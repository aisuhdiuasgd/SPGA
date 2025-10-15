

from __future__ import annotations
from typing import Tuple
from otherlayers import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from dataloader import GetData


class FeatureProjector(nn.Module):

    def __init__(self, in_dim: int, latent_dim: int, pca_dim: int = 128):
        super().__init__()
        self.pca_dim = pca_dim
        self.latent = nn.Linear(pca_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.latent(x)


# model.py —— 用这段替换掉占位版 MDI
class MDI(nn.Module):
    def __init__(self, latent_dim):
        super(MDI, self).__init__()
        self.inSize = latent_dim
        self.outSize = latent_dim
        self.gcnlayers = 1
        self.hdnDropout = 0.25
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()

        self.nodeGCN = GCN(self.inSize, self.outSize, dropout=self.hdnDropout,
                           layers=self.gcnlayers, resnet=True, act_fn=self.relu)
        self.layeratt_m = LayerAtt(self.inSize, self.outSize, self.gcnlayers)
        self.layeratt_d = LayerAtt(self.inSize, self.outSize, self.gcnlayers)
        self.fcLinear2 = ImprovedMLP(self.outSize, 1)

    def _norm_adj(self, A: torch.Tensor) -> torch.Tensor:

        deg = A.sum(dim=2)
        inv_sqrt = deg.clamp_min(1e-8).pow(-0.5)
        D = torch.zeros_like(A)
        idx = torch.arange(A.size(1), device=A.device)
        D[:, idx, idx] = inv_sqrt
        return D @ A @ D

    def forward(self, em: torch.Tensor, ed: torch.Tensor) -> torch.Tensor:

        device = em.device
        xm, xd = em.unsqueeze(1), ed.unsqueeze(1)
        node = torch.cat([xm, xd], dim=1)
        B, N, C = node.shape

        manh = torch.cdist(node, node, p=1)
        manh = self.lrelu(manh)
        idx = torch.arange(N, device=device)
        manh[:, idx, idx] = 1.0
        pL3 = self._norm_adj(manh)

        norm = (node.pow(2).sum(dim=2, keepdim=True).add(1e-8)).sqrt()
        cos = (node @ node.transpose(1, 2)) / (norm * norm.transpose(1, 2))
        cos = self.lrelu(cos)
        cos[:, idx, idx] = 1.0
        pL1 = self._norm_adj(cos)

        pL = torch.minimum(pL1, pL3)

        m_all, d_all = self.nodeGCN(node, pL)
        mLA = self.layeratt_m(m_all)
        dLA = self.layeratt_d(d_all)

        node_embed = mLA * dLA
        pre = self.fcLinear2(node_embed)
        return self.sigmoid(pre).squeeze(1)


class SPGAModel(nn.Module):

    def __init__(self, mdi: MDI, dataset: GetData, latent_dim: int, keep_prob: float = 0.8, dropout_bool: bool = False):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.use_dropout = bool(dropout_bool)
        self.keep_prob = float(keep_prob)


        self.num_snoRNAs = int(dataset.n_snoRNA)
        self.num_diseases = int(dataset.m_disease)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.md_supernode = mdi

        self.projector = FeatureProjector(in_dim=0, latent_dim=self.latent_dim, pca_dim=128)

        self.gate_s = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.gate_d = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

        self.dropout = nn.Dropout(p=1.0 - self.keep_prob)

        self.sigmoid = nn.Sigmoid()

    def computer(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        x_pca = PCA(n_components=128).fit_transform(x_np)

        x_t = torch.from_numpy(x_pca).float().to(self.device)

        all_emb = self.projector(x_t)

        # 切分
        snoRNAs, diseases = torch.split(all_emb, [self.num_snoRNAs, self.num_diseases], dim=0)
        return snoRNAs, diseases

    def forward(self, snoRNAs, disease, feature):

        self.train()

        s_emb, d_emb = self.computer(feature)  # (S, D)

        if self.use_dropout:
            s_emb = self.dropout(s_emb)
            d_emb = self.dropout(d_emb)

        if not torch.is_tensor(snoRNAs):
            snoRNAs = torch.as_tensor(snoRNAs, dtype=torch.long, device=self.device)
        if not torch.is_tensor(disease):
            disease = torch.as_tensor(disease, dtype=torch.long, device=self.device)

        s_batch = s_emb.index_select(0, snoRNAs)
        d_batch = d_emb.index_select(0, disease)

        pre_a = self.md_supernode(s_batch, d_batch)
        return pre_a


SPGA = SPGAModel
