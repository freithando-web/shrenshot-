import math
import random
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import json


@dataclass
class Tensor:
    data: List[List[float]]
    grad: List[List[float]] = field(default_factory=list)
    requires_grad: bool = False

    def __post_init__(self):
        if self.requires_grad and not self.grad:
            self.grad = [[0.0] * len(row) for row in self.data]

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.data), len(self.data[0]) if self.data else 0

    def zeros_grad(self):
        rows, cols = self.shape
        self.grad = [[0.0] * cols for _ in range(rows)]

    def __add__(self, other: 'Tensor') -> 'Tensor':
        assert self.shape == other.shape
        result = [[self.data[i][j] + other.data[i][j]
                   for j in range(self.shape[1])]
                  for i in range(self.shape[0])]
        return Tensor(result, requires_grad=self.requires_grad or other.requires_grad)

    def __mul__(self, scalar: float) -> 'Tensor':
        result = [[self.data[i][j] * scalar
                   for j in range(self.shape[1])]
                  for i in range(self.shape[0])]
        return Tensor(result, requires_grad=self.requires_grad)

    def T(self) -> 'Tensor':
        rows, cols = self.shape
        transposed = [[self.data[i][j] for i in range(rows)] for j in range(cols)]
        return Tensor(transposed)

    def matmul(self, other: 'Tensor') -> 'Tensor':
        m, n = self.shape
        n2, p = other.shape
        assert n == n2, f"Shape mismatch: {self.shape} vs {other.shape}"
        result = []
        for i in range(m):
            row = []
            for j in range(p):
                val = sum(self.data[i][k] * other.data[k][j] for k in range(n))
                row.append(val)
            result.append(row)
        return Tensor(result, requires_grad=self.requires_grad or other.requires_grad)

    @classmethod
    def randn(cls, rows: int, cols: int, std: float = 0.01) -> 'Tensor':
        data = [[random.gauss(0, std) for _ in range(cols)] for _ in range(rows)]
        return cls(data, requires_grad=True)

    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'Tensor':
        return cls([[0.0] * cols for _ in range(rows)], requires_grad=True)

    @classmethod
    def ones(cls, rows: int, cols: int) -> 'Tensor':
        return cls([[1.0] * cols for _ in range(rows)])


def relu(t: Tensor) -> Tensor:
    result = [[max(0.0, v) for v in row] for row in t.data]
    return Tensor(result, requires_grad=t.requires_grad)


def sigmoid(t: Tensor) -> Tensor:
    def _sig(x):
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)
    result = [[_sig(v) for v in row] for row in t.data]
    return Tensor(result, requires_grad=t.requires_grad)


def tanh_act(t: Tensor) -> Tensor:
    result = [[math.tanh(v) for v in row] for row in t.data]
    return Tensor(result, requires_grad=t.requires_grad)


def softmax(t: Tensor) -> Tensor:
    result = []
    for row in t.data:
        max_val = max(row)
        exps = [math.exp(v - max_val) for v in row]
        total = sum(exps)
        result.append([e / total for e in exps])
    return Tensor(result)


def layer_norm(t: Tensor, eps: float = 1e-5) -> Tensor:
    result = []
    for row in t.data:
        mean = sum(row) / len(row)
        variance = sum((v - mean) ** 2 for v in row) / len(row)
        std = math.sqrt(variance + eps)
        result.append([(v - mean) / std for v in row])
    return Tensor(result, requires_grad=t.requires_grad)


def dropout(t: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training:
        return t
    scale = 1.0 / (1.0 - p)
    result = [
        [v * scale if random.random() > p else 0.0 for v in row]
        for row in t.data
    ]
    return Tensor(result, requires_grad=t.requires_grad)


class LinearLayer:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weight = Tensor.randn(in_features, out_features,
                                   std=math.sqrt(2.0 / in_features))
        self.bias = Tensor.zeros(1, out_features) if bias else None
        self.in_features = in_features
        self.out_features = out_features
        self._input_cache: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        self._input_cache = x
        out = x.matmul(self.weight)
        if self.bias:
            rows, cols = out.shape
            result = [[out.data[i][j] + self.bias.data[0][j]
                       for j in range(cols)] for i in range(rows)]
            return Tensor(result, requires_grad=True)
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        params = [self.weight]
        if self.bias:
            params.append(self.bias)
        return params

    def num_parameters(self) -> int:
        rows, cols = self.weight.shape
        count = rows * cols
        if self.bias:
            count += self.bias.shape[1]
        return count


class AttentionHead:
    def __init__(self, embed_dim: int, head_dim: int):
        self.q_proj = LinearLayer(embed_dim, head_dim)
        self.k_proj = LinearLayer(embed_dim, head_dim)
        self.v_proj = LinearLayer(embed_dim, head_dim)
        self.scale = math.sqrt(head_dim)
        self.head_dim = head_dim

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores_data = q.matmul(k.T())
        scaled = [[v / self.scale for v in row] for row in scores_data.data]
        if mask:
            for i in range(len(scaled)):
                for j in range(len(scaled[i])):
                    if mask.data[i][j] == 0:
                        scaled[i][j] = -1e9
        attn_weights = softmax(Tensor(scaled))
        return attn_weights.matmul(v)

    def parameters(self) -> List[Tensor]:
        return (self.q_proj.parameters() +
                self.k_proj.parameters() +
                self.v_proj.parameters())


class MultiHeadAttention:
    def __init__(self, embed_dim: int, num_heads: int):
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.heads = [AttentionHead(embed_dim, self.head_dim) for _ in range(num_heads)]
        self.out_proj = LinearLayer(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        head_outputs = [head.forward(x) for head in self.heads]
        rows = head_outputs[0].shape[0]
        concat_data = []
        for i in range(rows):
            row = []
            for h in head_outputs:
                row.extend(h.data[i])
            concat_data.append(row)
        concat = Tensor(concat_data, requires_grad=True)
        return self.out_proj(concat)

    def parameters(self) -> List[Tensor]:
        params = []
        for head in self.heads:
            params.extend(head.parameters())
        params.extend(self.out_proj.parameters())
        return params


class FeedForward:
    def __init__(self, embed_dim: int, ff_dim: int, dropout_p: float = 0.1):
        self.fc1 = LinearLayer(embed_dim, ff_dim)
        self.fc2 = LinearLayer(ff_dim, embed_dim)
        self.dropout_p = dropout_p

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        h = relu(self.fc1(x))
        h = dropout(h, self.dropout_p, training)
        return self.fc2(h)

    def parameters(self) -> List[Tensor]:
        return self.fc1.parameters() + self.fc2.parameters()


class TransformerBlock:
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.embed_dim = embed_dim

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        attn_out = self.attn.forward(x)
        x_data = [[x.data[i][j] + attn_out.data[i][j]
                   for j in range(self.embed_dim)]
                  for i in range(x.shape[0])]
        x = layer_norm(Tensor(x_data, requires_grad=True))
        ff_out = self.ff.forward(x, training)
        x_data = [[x.data[i][j] + ff_out.data[i][j]
                   for j in range(self.embed_dim)]
                  for i in range(x.shape[0])]
        return layer_norm(Tensor(x_data, requires_grad=True))

    def parameters(self) -> List[Tensor]:
        return self.attn.parameters() + self.ff.parameters()


class MiniTransformer:
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_layers: int, ff_dim: int, max_seq_len: int = 512):
        self.embed = [[random.gauss(0, 0.02) for _ in range(embed_dim)]
                      for _ in range(vocab_size)]
        self.pos_embed = [[math.sin(pos / 10000 ** (2 * i / embed_dim))
                           if i % 2 == 0
                           else math.cos(pos / 10000 ** (2 * (i - 1) / embed_dim))
                           for i in range(embed_dim)]
                          for pos in range(max_seq_len)]
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_dim)
                       for _ in range(num_layers)]
        self.out_proj = LinearLayer(embed_dim, vocab_size)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    def forward(self, token_ids: List[int], training: bool = True) -> Tensor:
        seq_len = len(token_ids)
        x_data = [
            [self.embed[tid][j] + self.pos_embed[i][j]
             for j in range(self.embed_dim)]
            for i, tid in enumerate(token_ids)
        ]
        x = Tensor(x_data, requires_grad=True)
        for block in self.blocks:
            x = block.forward(x, training)
        logits = self.out_proj(x)
        return logits

    def generate(self, prompt: List[int], max_new_tokens: int = 50,
                 temperature: float = 1.0) -> List[int]:
        tokens = list(prompt)
        for _ in range(max_new_tokens):
            logits = self.forward(tokens[-512:], training=False)
            last_logits = logits.data[-1]
            if temperature != 1.0:
                last_logits = [v / temperature for v in last_logits]
            probs = softmax(Tensor([last_logits])).data[0]
            r = random.random()
            cumulative = 0.0
            next_token = 0
            for idx, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    next_token = idx
                    break
            tokens.append(next_token)
        return tokens[len(prompt):]

    def num_parameters(self) -> int:
        total = self.vocab_size * self.embed_dim
        for block in self.blocks:
            for p in block.parameters():
                rows, cols = p.shape
                total += rows * cols
        rows, cols = self.out_proj.weight.shape
        total += rows * cols
        return total


class SGD:
    def __init__(self, params: List[Tensor], lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = {id(p): Tensor.zeros(*p.shape) for p in params}

    def step(self) -> None:
        for p in self.params:
            if not p.grad:
                continue
            rows, cols = p.shape
            v = self._velocity[id(p)]
            for i in range(rows):
                for j in range(cols):
                    grad = p.grad[i][j]
                    if self.weight_decay:
                        grad += self.weight_decay * p.data[i][j]
                    if self.momentum:
                        v.data[i][j] = self.momentum * v.data[i][j] - self.lr * grad
                        p.data[i][j] += v.data[i][j]
                    else:
                        p.data[i][j] -= self.lr * grad

    def zero_grad(self) -> None:
        for p in self.params:
            p.zeros_grad()


def cross_entropy_loss(logits: Tensor, targets: List[int]) -> float:
    total_loss = 0.0
    probs = softmax(logits)
    for i, target in enumerate(targets):
        p = max(probs.data[i][target], 1e-12)
        total_loss -= math.log(p)
    return total_loss / len(targets)


def accuracy(logits: Tensor, targets: List[int]) -> float:
    correct = 0
    for i, target in enumerate(targets):
        pred = max(range(len(logits.data[i])), key=lambda j: logits.data[i][j])
        if pred == target:
            correct += 1
    return correct / len(targets)


if __name__ == '__main__':
    model = MiniTransformer(
        vocab_size=256,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        max_seq_len=128
    )
    print(f"Model parameters: {model.num_parameters():,}")
    tokens = [65, 66, 67, 68, 69]
    logits = model.forward(tokens)
    print(f"Output shape: {logits.shape}")
    loss = cross_entropy_loss(logits, [66, 67, 68, 69, 70])
    print(f"Loss: {loss:.4f}")
