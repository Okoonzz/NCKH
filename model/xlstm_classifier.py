import torch
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack, xLSTMBlockStackConfig,
    mLSTMBlockConfig, mLSTMLayerConfig,
    sLSTMBlockConfig, sLSTMLayerConfig,
    FeedForwardConfig
)

class xLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, context_length=50):
        super().__init__()
        # Embedding lookup (index -> vector), padding_idx=0 ensures PAD vector stays zero
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Cấu hình xLSTMBlockStack với mLSTM và sLSTM blocks
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=3,
                    qkv_proj_blocksize=4,
                    num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=4,
                    conv1d_kernel_size=3,
                    bias_init="powerlaw_blockdependent"
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=1.0,
                    act_fn="gelu"
                )
            ),
            context_length=context_length,
            num_blocks=3,
            embedding_dim=embed_dim,
            slstm_at=[1]
        )
        self.xlstm = xLSTMBlockStack(cfg)
        # Classifier từ vector ẩn cuối -> logit
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len) index tensor
        emb = self.embedding(x)            # (batch, seq_len, embed_dim)
        h = self.xlstm(emb)                # (batch, seq_len, embed_dim)
        last = h[:, -1, :]                 # (batch, embed_dim) lấy state cuối
        return self.classifier(last)      # (batch,1) logit