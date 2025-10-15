
from __future__ import annotations
from typing import Tuple, List

import torch
from torch import nn
from math import sqrt


class FeatureBatchNorm1d(nn.Module):

    def __init__(self, num_features: int, name: str = "batchNorm1d") -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.bn(x) if x.dim() == 2 else self.bn(x.transpose(-1, -2)).transpose(-1, -2)


class NodeEmbeddingWithDropout(nn.Module):

    def __init__(self, embedding: torch.Tensor, dropout: float, freeze: bool = False) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(embedding, dtype=torch.float32).detach(), freeze=freeze
        )
        self.dropout1 = nn.Dropout1d(p=dropout / 2)
        self.dropout2 = nn.Dropout(p=dropout / 2)
        self.p = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        return self.dropout2(self.dropout1(emb)) if self.p > 0 else emb



class LinearHead(nn.Module):

    def __init__(self, in_features: int, out_features: int, act_fn: nn.Module, use_bn: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.use_bn = use_bn
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        if self.use_bn:
            y = self.bn(y) if y.dim() == 2 else self.bn(y.transpose(-1, -2)).transpose(-1, -2)
        return y


class DeepLinearHead(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualGCN(nn.Module):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        dropout: float,
        layers: int,
        resnet: bool,
        act_fn: nn.Module,
        out_act: bool = True,
        out_dp: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = int(layers)
        self.act_fn = act_fn
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(in_size, out_size)
        self.use_act = bool(out_act)
        self.use_dp = bool(out_dp)
        self.use_res = bool(resnet)

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        z = x
        m_all = z[:, 0, :].unsqueeze(1)
        d_all = z[:, 1, :].unsqueeze(1)

        for _ in range(self.num_layers):
            a = torch.matmul(L, z)
            a = self.proj(a)
            if self.use_act:
                a = self.act_fn(a)
            if self.use_dp:
                a = self.dropout(a)
            if self.use_res and a.shape == z.shape:
                a = a + z
            z = a
            m_all = torch.cat((m_all, z[:, 0, :].unsqueeze(1)), dim=1)
            d_all = torch.cat((d_all, z[:, 1, :].unsqueeze(1)), dim=1)

        return m_all, d_all


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) 或 (B, C)
        if x.dim() == 3:
            s = x.mean(dim=1, keepdim=True)
        else:
            s = x.unsqueeze(1)

        s = self.fc1(s)
        s = torch.relu(s)
        s = self.fc2(s)
        g = self.sigmoid(s)

        return g.expand_as(x) * x




class MultiHeadLayerAttention(nn.Module):

    def __init__(self, in_size: int, out_size: int, gcnlayers: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.layers = int(gcnlayers) + 1
        self.in_size = int(in_size)
        self.out_size = int(out_size)

        self.q = nn.ModuleList([nn.Linear(in_size, out_size // num_heads) for _ in range(num_heads)])
        self.k = nn.ModuleList([nn.Linear(in_size, out_size // num_heads) for _ in range(num_heads)])
        self.v = nn.ModuleList([nn.Linear(in_size, out_size // num_heads) for _ in range(num_heads)])
        self.scale = 1.0 / sqrt(out_size)
        self.sm = nn.Softmax(dim=2)
        self.se = SEBlock(out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads: List[torch.Tensor] = []
        for i in range(self.num_heads):
            Q = self.q[i](x)
            K = self.k[i](x)
            V = self.v[i](x)
            attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, L, L)
            alpha = self.sm(attn)
            z = torch.bmm(alpha, V)
            heads.append(z)
        z = torch.cat(heads, dim=-1)
        z = self.se(z)
        return z.mean(dim=1)



BatchNorm1d = FeatureBatchNorm1d
BnodeEmbedding = NodeEmbeddingWithDropout
MLP = LinearHead
ImprovedMLP = DeepLinearHead
GCN = ResidualGCN
LayerAtt = MultiHeadLayerAttention


