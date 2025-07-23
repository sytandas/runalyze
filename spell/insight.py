'''
Experimental phi-2 to spell out some words about runs
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * (self.weight / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs

def apply_rotary(x, freqs):
    x1, x2 = x[..., ::2], x[..., 1::2]
    sin, cos = freqs.sin(), freqs.cos()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv.unbind(2)
        q = apply_rotary(q, freqs)
        k = apply_rotary(k, freqs)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(torch.triu(torch.ones(T, T, device=x.device), 1) == 1, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = 4 * dim
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x, freqs):
        x = x + self.attn(self.norm1(x), freqs)
        x = x + self.ff(self.norm2(x))
        return x

class Phi2Model(nn.Module):
    def __init__(self, vocab_size=51200, block_size=2048, dim=2560, n_layers=32, n_heads=32):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_emb = RotaryEmbedding(dim // n_heads)
        self.blocks = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.token_embed(input_ids)
        freqs = self.pos_emb(T, input_ids.device)
        for block in self.blocks:
            x = block(x, freqs)
        x = self.norm(x)
        return self.output(x)


def main():
    tokenizer = get_tokenizer()
    model = Phi2Model().cuda()
    load_phi2_weights(model)
    prompt = "Capital of India is "
    output = generate(model, tokenizer, prompt, max_new_tokens=1)
    print(output)

if __name__ == "__main__":
    main()

# TODO: algorithm to test 