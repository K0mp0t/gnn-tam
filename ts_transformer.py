import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, adj, X):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        h = self.dense(X)
        norm = adj.sum(1)**(-1/2)
        h = norm[None, :] * adj * norm[:, None] @ h
        return h

class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, theta=10000.0):
        super().__init__()
        self.d_head = d_model // nhead
        self.nhead = nhead
        self.theta = theta

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _apply_rope(self, Q, K):
        B, H, T, D = Q.shape
        qr, qi = Q.view(B, H, T, D//2, 2).unbind(-1)
        kr, ki = K.view(B, H, T, D//2, 2).unbind(-1)
        freqs = torch.arange(0, D//2, device=Q.device) * (1.0 / self.theta)
        angles = torch.einsum('t,d->t d', torch.arange(T, device=Q.device), freqs)
        cos = torch.cos(angles)[None, None, :, :]
        sin = torch.sin(angles)[None, None, :, :]
        qr2 = qr * cos - qi * sin
        qi2 = qr * sin + qi * cos
        kr2 = kr * cos - ki * sin
        ki2 = kr * sin + ki * cos
        Q2 = torch.stack([qr2, qi2], dim=-1).reshape(B, H, T, D)
        K2 = torch.stack([kr2, ki2], dim=-1).reshape(B, H, T, D)
        return Q2, K2

    def forward(self, x):
        B, T, d_model = x.size()
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(B, T, self.nhead, self.d_head).transpose(1,2)
        K = K.view(B, T, self.nhead, self.d_head).transpose(1,2)
        V = V.view(B, T, self.nhead, self.d_head).transpose(1,2)
        Q, K = self._apply_rope(Q, K)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1,2).contiguous().view(B, T, d_model)
        return self.out_proj(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.linear2(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttentionRoPE(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        att = self.attn(x)
        x = self.norm1(x + self.dropout(att))
        ff = self.ff(x)
        return self.norm2(x + self.dropout(ff))

class TimeSeriesTransformerGSL(nn.Module):
    def __init__(self, ts_dim, window_size, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1,
                 classes = 2,
                 device='cpu'):
        super().__init__()
        self.ts_dim = ts_dim
        self.window_size = window_size
        self.device = device
        self.input_proj = nn.Linear(ts_dim, d_model)

        self.layers = nn.ModuleList([
             TransformerBlock(d_model, nhead, dim_feedforward, dropout)
             for _ in range(num_layers)
         ])
        self.dropout = nn.Dropout(dropout)

        self.bnorm1 = nn.BatchNorm1d(d_model)
        self.fc = nn.Linear(d_model, d_model//2)
        self.fc_out = nn.Linear(d_model//2, classes)

    def forward(self, x):
        B, T, N = x.size()
        x_proj = self.input_proj(x)
        for layer in self.layers:
             x_proj = layer(x_proj)
    
        transformer_feat = x_proj.mean(dim=1)
        transformer_feat = self.bnorm1(transformer_feat)
        transformer_feat = self.dropout(transformer_feat)
        
        out = F.sigmoid(self.fc(transformer_feat))
        out = self.fc_out(out)
        return out