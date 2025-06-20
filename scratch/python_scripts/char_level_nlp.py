import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from tabulate import tabulate


# ---- Config ----
class Config(BaseModel):
    vocab_size: int = 100  # adjust this
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1


# ---- Positional Encoding ----
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, embed_dim, 2).float()
        angle_rates = 1 / torch.pow(10000, i / embed_dim)
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * angle_rates)
        pe[:, 1::2] = torch.cos(pos * angle_rates)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.device)


# ---- Transformer Block ----
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Linear(4 * embed_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x


# ---- Character-Level Transformer ----
class CharTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_enc = PositionalEncoding(cfg.embed_dim, cfg.max_seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.dropout)
                for _ in range(cfg.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size)

    def forward(self, x):
        B, T = x.size()
        tok_emb = self.token_emb(x)  # (B, T, D)
        x = self.pos_enc(tok_emb)  # (B, T, D)

        # Causal mask (prevent attending to future)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        mask = mask.masked_fill(mask, float("-inf"))

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)


# ---- Usage Example ----

if __name__ == "__main__":
    import logging

    def init_logger():
        """Initialize logger with a specific format."""
        logger = logging.getLogger(__name__)
        logger.handlers.clear()  # Ensure only one handler
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return logger

    logger = init_logger()
    logger.info("Starting character-level transformer script...")
    torch.manual_seed(42)
    logger.info("PyTorch seed set")

    cfg = Config(
        vocab_size=100, embed_dim=128, num_heads=4, num_layers=4, max_seq_len=128, dropout=0.1
    )

    logger.info(
        "Configuration initialized:\n"
        + tabulate(cfg.model_dump().items(), headers=["Parameter", "Value"], tablefmt="grid")
    )
    model = CharTransformer(cfg)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    x = torch.randint(0, cfg.vocab_size, (8, 32))  # batch of 8 sequences, 32 tokens each
    logger.info(f"Input tensor shape: {x.shape}")

    logits = model(x)
    logger.info(f"Output logits shape: {logits.shape}")  # (8, 32, vocab_size)
    logger.info("Script completed successfully!")
