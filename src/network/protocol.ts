import { EventEmitter } from 'events'
import * as crypto from 'crypto'
import * as zlib from 'zlib'
import { promisify } from 'util'

const gzip = promisify(zlib.gzip)
const gunzip = promisify(zlib.gunzip)

const PROTOCOL_VERSION = 0x0402
const MAX_PACKET_SIZE = 65536
const HEARTBEAT_INTERVAL = 5000
const CONNECTION_TIMEOUT = 30000
const MAX_RECONNECT_ATTEMPTS = 5

type PacketType =
  | 'HANDSHAKE'
  | 'HANDSHAKE_ACK'
  | 'DATA'
  | 'ACK'
  | 'HEARTBEAT'
  | 'HEARTBEAT_ACK'
  | 'DISCONNECT'
  | 'ERROR'
  | 'STREAM_INIT'
  | 'STREAM_DATA'
  | 'STREAM_END'

interface PacketHeader {
  version: number
  type: PacketType
  sequence: number
  timestamp: number
  sessionId: string
  checksum: string
  payloadSize: number
  flags: number
}

interface Packet {
  header: PacketHeader
  payload: Buffer
}

interface ConnectionOptions {
  host: string
  port: number
  tls?: boolean
  timeout?: number
  keepAlive?: boolean
  maxRetries?: number
  compressionThreshold?: number
  encryptionKey?: Buffer
}

interface SessionState {
  id: string
  remoteAddress: string
  connected: boolean
  authenticated: boolean
  sequence: number
  remoteSequence: number
  latency: number
  bytesSent: number
  bytesReceived: number
  packetsLost: number
  connectedAt: Date
  lastActivity: Date
}

class CircularBuffer {
  private buffer: Buffer
  private head = 0
  private tail = 0
  private size = 0

  constructor(private capacity: number) {
    this.buffer = Buffer.alloc(capacity)
  }

  write(data: Buffer): number {
    let written = 0
    for (let i = 0; i < data.length && this.size < this.capacity; i++) {
      this.buffer[this.tail] = data[i]
      this.tail = (this.tail + 1) % this.capacity
      this.size++
      written++
    }
    return written
  }

  read(n: number): Buffer {
    const count = Math.min(n, this.size)
    const result = Buffer.alloc(count)
    for (let i = 0; i < count; i++) {
      result[i] = this.buffer[this.head]
      this.head = (this.head + 1) % this.capacity
      this.size--
    }
    return result
  }

  get available(): number { return this.size }
  get free(): number { return this.capacity - this.size }
}

class PacketSerializer {
  static readonly HEADER_SIZE = 128

  static serialize(packet: Packet): Buffer {
    const headerBuf = Buffer.alloc(this.HEADER_SIZE)
    headerBuf.writeUInt16BE(packet.header.version, 0)
    headerBuf.writeUInt8(this.typeToCode(packet.header.type), 2)
    headerBuf.writeUInt32BE(packet.header.sequence, 3)
    headerBuf.writeDoubleBE(packet.header.timestamp, 7)
    Buffer.from(packet.header.sessionId.replace(/-/g, ''), 'hex').copy(headerBuf, 15)
    Buffer.from(packet.header.checksum, 'hex').copy(headerBuf, 31)
    headerBuf.writeUInt32BE(packet.header.payloadSize, 63)
    headerBuf.writeUInt16BE(packet.header.flags, 67)
    return Buffer.concat([headerBuf, packet.payload])
  }

  static deserialize(data: Buffer): Packet {
    if (data.length < this.HEADER_SIZE) throw new Error('Packet too short')
    const version = data.readUInt16BE(0)
    const typeCode = data.readUInt8(2)
    const sequence = data.readUInt32BE(3)
    const timestamp = data.readDoubleBE(7)
    const sessionId = this.bufToUuid(data.slice(15, 31))
    const checksum = data.slice(31, 63).toString('hex')
    const payloadSize = data.readUInt32BE(63)
    const flags = data.readUInt16BE(67)
    const payload = data.slice(this.HEADER_SIZE, this.HEADER_SIZE + payloadSize)
    if (payload.length !== payloadSize) throw new Error('Payload size mismatch')
    const actualChecksum = crypto.createHash('sha256').update(payload).digest('hex')
    if (actualChecksum !== checksum) throw new Error('Checksum verification failed')
    return {
      header: { version, type: this.codeToType(typeCode), sequence, timestamp,
                sessionId, checksum, payloadSize, flags },
      payload
    }
  }

  private static typeToCode(type: PacketType): number {
    const map: Record<PacketType, number> = {
      HANDSHAKE: 0x01, HANDSHAKE_ACK: 0x02, DATA: 0x10, ACK: 0x11,
      HEARTBEAT: 0x20, HEARTBEAT_ACK: 0x21, DISCONNECT: 0x30,
      ERROR: 0x40, STREAM_INIT: 0x50, STREAM_DATA: 0x51, STREAM_END: 0x52
    }
    return map[type] ?? 0xFF
  }

  private static codeToType(code: number): PacketType {
    const map: Record<number, PacketType> = {
      0x01: 'HANDSHAKE', 0x02: 'HANDSHAKE_ACK', 0x10: 'DATA', 0x11: 'ACK',
      0x20: 'HEARTBEAT', 0x21: 'HEARTBEAT_ACK', 0x30: 'DISCONNECT',
      0x40: 'ERROR', 0x50: 'STREAM_INIT', 0x51: 'STREAM_DATA', 0x52: 'STREAM_END'
    }
    return map[code] ?? 'ERROR'
  }

  private static bufToUuid(buf: Buffer): string {
    const hex = buf.toString('hex')
    return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`
  }
}

class RetransmissionQueue {
  private pending = new Map<number, { packet: Packet; sentAt: number; attempts: number }>()
  private rto = 200
  private srtt = 0
  private rttvar = 0
  private readonly MAX_ATTEMPTS = 5

  enqueue(packet: Packet): void {
    this.pending.set(packet.header.sequence, { packet, sentAt: Date.now(), attempts: 1 })
  }

  acknowledge(sequence: number): void {
    const entry = this.pending.get(sequence)
    if (entry) {
      const rtt = Date.now() - entry.sentAt
      this.updateRTT(rtt)
      this.pending.delete(sequence)
    }
  }

  getExpired(): Packet[] {
    const now = Date.now()
    const expired: Packet[] = []
    for (const [seq, entry] of this.pending) {
      if (now - entry.sentAt > this.rto) {
        if (entry.attempts >= this.MAX_ATTEMPTS) {
          this.pending.delete(seq)
        } else {
          entry.attempts++
          entry.sentAt = now
          expired.push(entry.packet)
        }
      }
    }
    return expired
  }

  private updateRTT(rtt: number): void {
    if (this.srtt === 0) {
      this.srtt = rtt
      this.rttvar = rtt / 2
    } else {
      this.rttvar = 0.75 * this.rttvar + 0.25 * Math.abs(this.srtt - rtt)
      this.srtt = 0.875 * this.srtt + 0.125 * rtt
    }
    this.rto = Math.max(200, Math.min(this.srtt + 4 * this.rttvar, 60000))
  }

  get pendingCount(): number { return this.pending.size }
  get estimatedRTT(): number { return this.srtt }
}

class FlowController {
  private windowSize: number
  private inflightBytes: number = 0
  private readonly initialWindow: number

  constructor(initialWindowKB: number = 64) {
    this.windowSize = initialWindowKB * 1024
    this.initialWindow = this.windowSize
  }

  canSend(bytes: number): boolean {
    return this.inflightBytes + bytes <= this.windowSize
  }

  onSend(bytes: number): void { this.inflightBytes += bytes }
  onAck(bytes: number): void {
    this.inflightBytes = Math.max(0, this.inflightBytes - bytes)
    this.windowSize = Math.min(this.windowSize * 1.5, 1024 * 1024)
  }
  onLoss(): void {
    this.windowSize = Math.max(this.windowSize * 0.5, this.initialWindow)
  }

  get currentWindow(): number { return this.windowSize }
  get utilization(): number { return this.inflightBytes / this.windowSize }
}

export class Connection extends EventEmitter {
  private session: SessionState
  private receiveBuffer: CircularBuffer
  private retransmitQueue: RetransmissionQueue
  private flowCtrl: FlowController
  private heartbeatTimer?: NodeJS.Timer
  private timeoutTimer?: NodeJS.Timer
  private encKey?: Buffer
  private cipher?: crypto.Cipher
  private decipher?: crypto.Decipher

  constructor(private options: ConnectionOptions) {
    super()
    this.session = {
      id: crypto.randomUUID(),
      remoteAddress: `${options.host}:${options.port}`,
      connected: false,
      authenticated: false,
      sequence: 0,
      remoteSequence: 0,
      latency: 0,
      bytesSent: 0,
      bytesReceived: 0,
      packetsLost: 0,
      connectedAt: new Date(),
      lastActivity: new Date(),
    }
    this.receiveBuffer = new CircularBuffer(MAX_PACKET_SIZE * 4)
    this.retransmitQueue = new RetransmissionQueue()
    this.flowCtrl = new FlowController()
    if (options.encryptionKey) this.setupEncryption(options.encryptionKey)
  }

  private setupEncryption(key: Buffer): void {
    this.encKey = key
    const iv = crypto.randomBytes(16)
    this.cipher = crypto.createCipheriv('aes-256-gcm', key, iv)
    this.decipher = crypto.createDecipheriv('aes-256-gcm', key, iv)
  }

  private buildPacket(type: PacketType, payload: Buffer, flags = 0): Packet {
    const checksum = crypto.createHash('sha256').update(payload).digest('hex')
    return {
      header: {
        version: PROTOCOL_VERSION,
        type,
        sequence: this.session.sequence++,
        timestamp: Date.now(),
        sessionId: this.session.id,
        checksum,
        payloadSize: payload.length,
        flags,
      },
      payload,
    }
  }

  async send(data: Buffer | string): Promise<void> {
    const buf = Buffer.isBuffer(data) ? data : Buffer.from(data)
    let payload = buf
    if (this.options.compressionThreshold && buf.length >= this.options.compressionThreshold) {
      payload = await gzip(buf)
    }
    if (this.cipher) {
      payload = Buffer.concat([this.cipher.update(payload), this.cipher.final()])
    }
    if (!this.flowCtrl.canSend(payload.length)) {
      await this.waitForWindow()
    }
    const packet = this.buildPacket('DATA', payload)
    const serialized = PacketSerializer.serialize(packet)
    this.retransmitQueue.enqueue(packet)
    this.flowCtrl.onSend(serialized.length)
    this.session.bytesSent += serialized.length
    this.session.lastActivity = new Date()
    this.emit('packet:sent', packet)
  }

  private waitForWindow(): Promise<void> {
    return new Promise(resolve => {
      const check = () => {
        if (this.flowCtrl.canSend(1)) resolve()
        else setTimeout(check, 10)
      }
      check()
    })
  }

  receive(data: Buffer): void {
    this.receiveBuffer.write(data)
    this.session.bytesReceived += data.length
    this.session.lastActivity = new Date()
    this.processBuffer()
  }

  private processBuffer(): void {
    while (this.receiveBuffer.available >= PacketSerializer.HEADER_SIZE) {
      try {
        const headerBytes = this.receiveBuffer.read(PacketSerializer.HEADER_SIZE)
        const payloadSize = headerBytes.readUInt32BE(63)
        if (this.receiveBuffer.available < payloadSize) {
          this.receiveBuffer.write(headerBytes)
          break
        }
        const payload = this.receiveBuffer.read(payloadSize)
        const fullPacket = Buffer.concat([headerBytes, payload])
        const packet = PacketSerializer.deserialize(fullPacket)
        this.handlePacket(packet)
      } catch (err) {
        this.emit('error', err)
        break
      }
    }
  }

  private handlePacket(packet: Packet): void {
    switch (packet.header.type) {
      case 'DATA':
        this.retransmitQueue.acknowledge(packet.header.sequence - 1)
        this.flowCtrl.onAck(packet.payload.length)
        this.emit('data', packet.payload)
        this.sendAck(packet.header.sequence)
        break
      case 'ACK':
        this.retransmitQueue.acknowledge(packet.header.sequence)
        break
      case 'HEARTBEAT':
        this.sendHeartbeatAck(packet.header.timestamp)
        break
      case 'HEARTBEAT_ACK':
        this.session.latency = Date.now() - packet.header.timestamp
        break
      case 'DISCONNECT':
        this.session.connected = false
        this.emit('disconnect', { reason: packet.payload.toString() })
        break
      case 'ERROR':
        this.emit('remote-error', { message: packet.payload.toString() })
        break
    }
  }

  private sendAck(seq: number): void {
    const ack = this.buildPacket('ACK', Buffer.from(seq.toString()))
    this.emit('packet:sent', ack)
  }

  private sendHeartbeatAck(originalTs: number): void {
    const pkt = this.buildPacket('HEARTBEAT_ACK', Buffer.from(originalTs.toString()))
    this.emit('packet:sent', pkt)
  }

  startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      const pkt = this.buildPacket('HEARTBEAT', Buffer.from(Date.now().toString()))
      this.emit('packet:sent', pkt)
    }, HEARTBEAT_INTERVAL)
  }

  stopHeartbeat(): void {
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer)
  }

  disconnect(reason = 'graceful'): void {
    const pkt = this.buildPacket('DISCONNECT', Buffer.from(reason))
    this.emit('packet:sent', pkt)
    this.stopHeartbeat()
    this.session.connected = false
  }

  get stats(): Readonly<SessionState> { return { ...this.session } }
  get pendingAcks(): number { return this.retransmitQueue.pendingCount }
  get estimatedLatency(): number { return this.retransmitQueue.estimatedRTT }
  get windowUtilization(): number { return this.flowCtrl.utilization }
}

export class ConnectionPool extends EventEmitter {
  private connections: Map<string, Connection> = new Map()
  private availableIds: string[] = []
  private waiters: Array<(conn: Connection) => void> = []

  constructor(
    private options: ConnectionOptions,
    private readonly maxSize: number = 10,
    private readonly minSize: number = 2,
  ) {
    super()
    for (let i = 0; i < minSize; i++) this.addConnection()
  }

  private addConnection(): Connection {
    const conn = new Connection(this.options)
    this.connections.set(conn.stats.id, conn)
    this.availableIds.push(conn.stats.id)
    conn.on('disconnect', () => this.removeConnection(conn.stats.id))
    return conn
  }

  private removeConnection(id: string): void {
    this.connections.delete(id)
    this.availableIds = this.availableIds.filter(i => i !== id)
    if (this.connections.size < this.minSize) this.addConnection()
  }

  async acquire(timeoutMs = 5000): Promise<Connection> {
    if (this.availableIds.length > 0) {
      const id = this.availableIds.pop()!
      return this.connections.get(id)!
    }
    if (this.connections.size < this.maxSize) {
      return this.addConnection()
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Pool timeout')), timeoutMs)
      this.waiters.push(conn => { clearTimeout(timer); resolve(conn) })
    })
  }

  release(conn: Connection): void {
    if (this.waiters.length > 0) {
      const waiter = this.waiters.shift()!
      waiter(conn)
    } else {
      this.availableIds.push(conn.stats.id)
    }
  }

  async withConnection<T>(fn: (conn: Connection) => Promise<T>): Promise<T> {
    const conn = await this.acquire()
    try {
      return await fn(conn)
    } finally {
      this.release(conn)
    }
  }

  get size(): number { return this.connections.size }
  get available(): number { return this.availableIds.length }
  get waiting(): number { return this.waiters.length }
}
