"""Small decoder-only transformer, implemented directly with PyTorch."""

from __future__ import annotations

from collections.abc import Callable

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise RuntimeError("Install the model extra with `pip install -e '.[model]'`.") from exc

from .tokenizer import BASE_VOCAB_SIZE
from .tokenizer import EOS, PAD


class CausalAttention(nn.Module):
    def __init__(self, width: int, heads: int, context: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(context, context)), persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t):
            return t.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(-2, -1)) * (self.head_width ** -0.5)
        scores = scores.masked_fill(self.mask[:length, :length] == 0, float("-inf"))
        weights = scores.softmax(dim=-1)
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended)


class Block(nn.Module):
    def __init__(self, width: int, heads: int, context: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = CausalAttention(width, heads, context)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ScratchTransformer(nn.Module):
    """A small CPU-oriented GPT-style model; all weights start randomly initialized."""

    def __init__(self, *, context: int = 256, width: int = 128, layers: int = 4, heads: int = 4, vocab_size: int = BASE_VOCAB_SIZE) -> None:
        super().__init__()
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*(Block(width, heads, context) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        # GPT-2 style residual scaling: without it the first forward pass emits
        # logits in the tens, the initial loss is ~88 instead of ~ln(vocab), and
        # a small model spends thousands of steps just shrinking its own output.
        scale = (2 * layers) ** -0.5
        for name, parameter in self.named_parameters():
            if name.endswith("output.weight") or name.endswith("mlp.2.weight"):
                parameter.data.mul_(scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None, loss_mask=None):
        _, length = tokens.shape
        if length > self.context:
            raise ValueError(f"sequence length {length} exceeds context {self.context}")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        logits = self.lm_head(self.norm(self.blocks(hidden)))
        loss = None
        if targets is not None:
            flat = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
            )
            if loss_mask is None:
                loss = flat.mean()
            else:
                weights = loss_mask.reshape(-1).to(flat.dtype)
                loss = (flat * weights).sum() / weights.sum().clamp(min=1.0)
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        *,
        max_new_tokens: int = 160,
        temperature: float = 0.8,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        should_stop: Callable[[], bool] | None = None,
    ):
        generated: list[int] = []
        for _ in range(max_new_tokens):
            # The owner stop flag is checked between autoregressive tokens. A
            # tool already executing is allowed to finish its bounded operation;
            # the agent checks the flag again before any later tool or model call.
            if should_stop is not None and should_stop():
                break
            logits, _ = self(tokens[:, -self.context:])
            next_logits = logits[:, -1, :]
            next_logits[:, PAD] = float("-inf")
            if repetition_penalty and repetition_penalty > 1.0 and generated:
                # Small models loop on their own output ("the the the"); scaling
                # the logits of tokens already emitted is the cheapest cure.
                for token in set(generated):
                    if next_logits[0, token] > 0:
                        next_logits[0, token] /= repetition_penalty
                    else:
                        next_logits[0, token] *= repetition_penalty
            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = (next_logits / temperature).softmax(dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_index = torch.sort(probs, descending=True, dim=-1)
                    cumulative = sorted_probs.cumsum(dim=-1)
                    keep = cumulative - sorted_probs < top_p
                    keep[..., 0] = True
                    filtered = torch.zeros_like(probs)
                    filtered.scatter_(1, sorted_index, sorted_probs * keep)
                    probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                next_token = torch.multinomial(probs, num_samples=1)
            generated.append(int(next_token[0, 0]))
            tokens = torch.cat((tokens, next_token), dim=1)
            if bool(torch.all(next_token == EOS)):
                break
        return tokens
