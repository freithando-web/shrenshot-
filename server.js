const express = require('express');
const Database = require('better-sqlite3');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const cookieParser = require('cookie-parser');
const path = require('path');

const app = express();
const db = new Database('./charity.db');
const JWT_SECRET = 'charity_secret_key_2024';

// Setup database
db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS donations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    user_name TEXT NOT NULL,
    amount REAL NOT NULL,
    message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
  );
`);

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

function requireAuth(req, res, next) {
  const token = req.cookies.token;
  if (!token) return res.status(401).json({ error: 'Not logged in' });
  try {
    req.user = jwt.verify(token, JWT_SECRET);
    next();
  } catch {
    res.status(401).json({ error: 'Invalid session' });
  }
}

// Register
app.post('/api/register', async (req, res) => {
  const { name, email, password } = req.body;
  if (!name || !email || !password) return res.status(400).json({ error: 'All fields required' });
  if (password.length < 6) return res.status(400).json({ error: 'Password must be at least 6 characters' });
  const hashed = await bcrypt.hash(password, 10);
  try {
    const stmt = db.prepare('INSERT INTO users (name, email, password) VALUES (?, ?, ?)');
    const result = stmt.run(name, email.toLowerCase(), hashed);
    const token = jwt.sign({ id: result.lastInsertRowid, name, email }, JWT_SECRET, { expiresIn: '7d' });
    res.cookie('token', token, { httpOnly: true, maxAge: 7 * 24 * 60 * 60 * 1000 });
    res.json({ success: true, name });
  } catch {
    res.status(400).json({ error: 'Email already registered' });
  }
});

// Login
app.post('/api/login', async (req, res) => {
  const { email, password } = req.body;
  const user = db.prepare('SELECT * FROM users WHERE email = ?').get(email.toLowerCase());
  if (!user) return res.status(401).json({ error: 'Invalid email or password' });
  const valid = await bcrypt.compare(password, user.password);
  if (!valid) return res.status(401).json({ error: 'Invalid email or password' });
  const token = jwt.sign({ id: user.id, name: user.name, email: user.email }, JWT_SECRET, { expiresIn: '7d' });
  res.cookie('token', token, { httpOnly: true, maxAge: 7 * 24 * 60 * 60 * 1000 });
  res.json({ success: true, name: user.name });
});

// Logout
app.post('/api/logout', (req, res) => {
  res.clearCookie('token');
  res.json({ success: true });
});

// Get current user
app.get('/api/me', requireAuth, (req, res) => {
  res.json({ name: req.user.name, email: req.user.email });
});

// Donate
app.post('/api/donate', requireAuth, (req, res) => {
  const { amount, message } = req.body;
  const num = parseFloat(amount);
  if (!num || num < 1) return res.status(400).json({ error: 'Minimum donation is $1' });
  if (num > 50000) return res.status(400).json({ error: 'Maximum single donation is $50,000' });
  db.prepare('INSERT INTO donations (user_id, user_name, amount, message) VALUES (?, ?, ?, ?)').run(
    req.user.id, req.user.name, num, message || null
  );
  res.json({ success: true });
});

// Get donation stats
app.get('/api/stats', (req, res) => {
  const total = db.prepare('SELECT COALESCE(SUM(amount), 0) as total FROM donations').get().total;
  const count = db.prepare('SELECT COUNT(*) as count FROM donations').get().count;
  const recent = db.prepare('SELECT user_name, amount, message, created_at FROM donations ORDER BY created_at DESC LIMIT 10').all();
  res.json({ total, count, recent });
});

app.listen(3000, () => console.log('Charity site running on http://localhost:3000'));
