# -*- coding: utf-8 -*-
"""
GRACE 訓練（Node-level 對比學習，含 Validation Probe）
- 兩個增廣 view（edge drop + feature mask），以 InfoNCE 進行 node-level 對比
- 曲線：Loss + Probe(AUC)，可用 Probe 作為早停與選模
- 適配 10GB GPU：支援梯度累積、AMP、對比損失分塊計算（避免 O(N^2) 記憶體）
"""

import argparse
import os
import sys
import json
import time
import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

# 依你現有專案
from graph_builder import build_graph_from_case, collect_all_cell_types
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau, StepLR

# ---------------------------
# 全域快取（固定驗證抽樣用）
# ---------------------------
PROBE_CACHE = {}


def ensure_cpu_data(data):
    """確保 Data 在 CPU（避免 DataLoader collation device 混用）"""
    return data.to('cpu')


# ---------------------------
# 參數
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description='GRACE（node-level 對比學習）訓練框架')

    # 模型架構（針對低顯存優化）
    p.add_argument('--hidden-dim', type=int, default=64, help='GAT 隱藏維度') 
    p.add_argument('--num-layers', type=int, default=2, help='GAT 層數')  # 保持 2 層
    p.add_argument('--dropout', type=float, default=0.3, help='GAT Dropout')  # 增加 dropout 防止過擬合
    p.add_argument('--cell-embed-dim', type=int, default=8, help='cell embedding 維度')  # 從 16 降至 8
    p.add_argument('--heads-schedule', type=str, default='1,1', help='每層 head 數，如 "1,1"；空字串用預設')  # 更保守的單頭配置

    # GRACE 超參（低顯存優化）
    p.add_argument('--tau', type=float, default=0.3, help='InfoNCE 溫度參數')  # 從 0.2 調整到 0.3  
    p.add_argument('--feat-mask', type=float, default=0.1, help='特徵隨機遮蔽比例（不遮最後一維 cell_id）')  # 從 0.15 降至 0.1
    p.add_argument('--edge-drop', type=float, default=0.1, help='邊隨機丟棄比例')  # 從 0.15 降至 0.1
    p.add_argument('--use-proj', action='store_true', help='啟用 projection head（2-layer MLP）')
    p.add_argument('--proj-dim', type=int, default=32, help='projection head 輸出維度')  # 從 64 降至 32

    # 訓練（低顯存優化）
    p.add_argument('--epochs', type=int, default=250)  
    p.add_argument('--lr', type=float, default=1e-3)  # 從 5e-4 提升至 1e-3，加快收斂
    p.add_argument('--weight-decay', type=float, default=5e-4)  # 從 1e-4 增至 5e-4，增強正則化
    p.add_argument('--batch-size', type=int, default=1)  # 強制設為 1
    p.add_argument('--gradient-accumulation', type=int, default=8)  # 從 4 增至 8
    p.add_argument('--grad-clip', type=float, default=0.5)  # 從 1.0 降至 0.5，更保守的梯度裁剪
    p.add_argument('--warmup-epochs', type=int, default=5, help='學習率預熱 epochs')  # 從 10 降至 5

    # 對比損失計算參數（省顯存）
    p.add_argument('--nce-chunk', type=int, default=128, help='InfoNCE 分塊大小（行數）')  # 從 4096 大幅降至 128

    # 早停/調度
    p.add_argument('--patience', type=int, default=50)  # 進一步增加耐心，避免過早停止
    p.add_argument('--scheduler-factor', type=float, default=0.8)  # 更溫和的LR衰減
    p.add_argument('--scheduler-patience', type=int, default=25)  # 大幅延後LR調度時機
    p.add_argument('--scheduler-type', type=str, default='cosine', choices=['cosine', 'plateau', 'step'],
                   help='學習率調度器類型: cosine=余弦退火, plateau=基於loss, step=固定步長')

    # 輸出
    p.add_argument('--print-every', type=int, default=10)
    p.add_argument('--save-best', action='store_true')
    p.add_argument('--output-name', type=str, default='encoder_grace.pt')

    # 資料
    p.add_argument('--benchmarks', type=str, default='c17,c432,c499,c880,c1355')
    p.add_argument('--test-only', action='store_true', help='僅用 s27，快速測試')
    p.add_argument('--separate-training', action='store_true', help='逐圖訓練')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-meta', action='store_true')

    # Probe（驗證）
    p.add_argument('--probe-every', type=int, default=10)
    p.add_argument('--probe-samples', type=int, default=30000)
    p.add_argument('--earlystop-by-probe', action='store_true')
    p.add_argument('--probe-graph', type=str, default='', help='指定哪個基準圖作為 probe')
    p.add_argument('--fix-probe-samples', action='store_true', help='固定 probe 抽樣（更穩定）')
    p.add_argument('--disable-probe', action='store_true', help='完全關閉 probe 驗證（純訓練模式）')
    p.add_argument('--final-probe-average', type=int, default=0, help='訓練後重複抽樣回報平均 AUC 次數 (0=關閉)')

    # 裝置
    p.add_argument('--device', type=str, default='auto', choices=['auto','cuda','cpu'])
    p.add_argument('--disable-auto-tune', action='store_true', help='關閉自動調參（batch/nce-chunk）')
    return p.parse_args()


# ---------------------------
# GAT Encoder + Projection
# ---------------------------

class ConfigurableGATEncoder(nn.Module):
    """可配置 GAT 編碼器（附 cell embedding，concat=False 省顯存）"""
    def __init__(self, in_channels, out_channels, num_layers=2, heads_schedule=None, dropout=0.1,
                 cell_embedding_dim=16, num_cells=854):
        super().__init__()
        self.cell_embedding = nn.Embedding(num_cells, cell_embedding_dim)
        actual_in_channels = in_channels - 1 + cell_embedding_dim  # -1: 去除 cell_id

        # heads 配置
        if not heads_schedule:
            if num_layers == 1:
                heads_schedule = [1]
            elif num_layers == 2:
                heads_schedule = [2, 1]
            elif num_layers == 3:
                heads_schedule = [2, 2, 1]  # 更保守的配置
            else:
                heads_schedule = [2] * (num_layers - 1) + [1]

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        in_dim = actual_in_channels
        for i in range(num_layers):
            heads = heads_schedule[i] if i < len(heads_schedule) else 1
            self.convs.append(GATConv(in_dim, out_channels, heads=heads, concat=False, dropout=dropout))
            if i < num_layers - 1:
                self.bns.append(nn.LayerNorm(out_channels))
            in_dim = out_channels
        self.dropout = dropout

    def forward(self, x, edge_index):
        base = x[:, :-1]
        cell_ids = x[:, -1].long()
        cell_emb = self.cell_embedding(cell_ids)
        x = torch.cat([base, cell_emb], dim=1)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 簡化投影頭以節省記憶體
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)



# ---------------------------
# Augmentation（不丟 node）
# ---------------------------

def augment_view(x, edge_index, feat_mask_p=0.15, edge_drop_p=0.15):
    """建立單一 view：遮蔽部分特徵、隨機移除邊
    - 不改變節點數與順序（避免正對應破壞）
    - 最後一維（cell_id）不做遮蔽
    - 簡化版，節省記憶體
    """
    device = x.device
    x1 = x.clone()
    
    # 特徵增廣：只做基本遮蔽
    if feat_mask_p > 0:
        if x1.size(1) > 1:
            mask = (torch.rand_like(x1[:, :-1]) < feat_mask_p)
            x1[:, :-1][mask] = 0
    
    # 邊增廣：只做隨機丟棄
    e1 = edge_index
    if edge_drop_p > 0 and edge_index.numel() > 0:
        E = edge_index.size(1)
        keep = torch.rand(E, device=device) > edge_drop_p
        e1 = edge_index[:, keep]
    
    return x1, e1


# ---------------------------
# InfoNCE（分塊版，節省記憶體）
# ---------------------------

def nt_xent_loss_chunked(z1, z2, tau=0.5, chunk=256):
    # L2 normalize
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)

    N, D = z1.size()
    assert z2.size(0) == N and z2.size(1) == D
    device = z1.device
    
    # 進一步減小 chunk 大小以節省記憶體
    chunk = min(chunk, N)

    def _dir_loss(a, b):
        # 計算 -log( exp(sim_pos)/sum_j exp(sim_all) )，行方向分塊平均
        total = torch.zeros((), device=device)
        count = 0
        bt = b.t()  # [D, N]
        for i0 in range(0, N, chunk):
            i1 = min(i0 + chunk, N)
            z = a[i0:i1]                            # [B, D]
            logits = (z @ bt).float() / tau                 # [B, N]
            idx = torch.arange(i0, i1, device=device).view(-1, 1)
            pos = logits.gather(1, idx).squeeze(1)  # [B]
            lse = torch.logsumexp(logits, dim=1)    # [B]
            # 累加總和（保持為 Tensor）
            total = total + (-pos + lse).sum()
            count += (i1 - i0)
            
            # 立即清理中間變數
            del z, logits, idx, pos, lse
            
        return total / max(1, count)

    loss1 = _dir_loss(z1, z2)
    loss2 = _dir_loss(z2, z1)
    return 0.5 * (loss1 + loss2)  # torch.Tensor



# ---------------------------
# Validation：Link Prediction AUC（與你原版一致）
# ---------------------------
@torch.no_grad()
def link_pred_probe_auc(encoder, data, device, num_samples=2000, cache_key=None, fix_probe_samples=False, proj=None):
    encoder.eval()
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    use_amp = (device.type == 'cuda')
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
        z = encoder(x, edge_index)
        if proj is not None:
            z = proj(z)
    z = F.normalize(z, p=2, dim=-1)

    # 取得/建立抽樣
    pos = neg = None
    if fix_probe_samples and (cache_key is not None) and (cache_key in PROBE_CACHE):
        pos, neg = PROBE_CACHE[cache_key]

    if (pos is None) or (neg is None):
        E = edge_index.size(1)
        take = min(num_samples, E)
        idx = torch.randperm(E, device=device)[:take]
        pos = edge_index[:, idx]
        
        # 生成更難的負樣本：優先選擇 cell_id 相同的負樣本
        cand = negative_sampling(edge_index=edge_index, num_nodes=x.size(0), num_neg_samples=take*4)
        cell_ids = x[:, -1].long()
        mask = (cell_ids[cand[0]] == cell_ids[cand[1]])
        if mask.sum() >= take:
            neg = cand[:, mask][:, :take]
        else:
            neg = negative_sampling(edge_index=edge_index, num_nodes=x.size(0), num_neg_samples=take)
            
        if fix_probe_samples and (cache_key is not None):
            PROBE_CACHE[cache_key] = (pos, neg)

    # 分批打分
    def _batched_score(pairs, z, chunk=2048):
        outs = []
        for i in range(0, pairs.size(1), chunk):
            a = pairs[0, i:i+chunk]
            b = pairs[1, i:i+chunk]
            outs.append((z[a] * z[b]).sum(dim=-1))
        return torch.cat(outs, dim=0)

    pos_s = _batched_score(pos, z, chunk=2048)
    neg_s = _batched_score(neg, z, chunk=2048)

    scores = torch.cat([pos_s, neg_s], dim=0)
    labels = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)], dim=0)

    order = torch.argsort(scores, descending=True)
    y = labels[order].float()
    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1 - y, dim=0)
    P, N = y.sum(), (1 - y).sum()
    if P == 0 or N == 0:
        return 0.5
    tpr, fpr = tp / P, fp / N
    auc = torch.trapz(tpr, fpr).item()
    return auc


@torch.no_grad()
def recall_at_k(encoder, data, device, K=10, proj=None):
    """計算 Recall@K 指標"""
    encoder.eval()
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    
    use_amp = (device.type == 'cuda')
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
        z = encoder(x, edge_index)
        if proj is not None:
            z = proj(z)
    z = F.normalize(z, p=2, dim=-1)

    N = z.size(0)
    A = torch.zeros((N, N), dtype=torch.bool, device=device)
    A[edge_index[0], edge_index[1]] = True
    A[edge_index[1], edge_index[0]] = True     # 無向圖可留著

    S = z @ z.t()
    S.fill_diagonal_(-float('inf'))
    topk = torch.topk(S, k=min(K, N-1), dim=1).indices
    hits = A.gather(1, topk).float().sum(dim=1)
    deg = A.sum(dim=1).clamp_min(1).float()
    return (hits / deg).mean().item()


# ---------------------------
# 訓練主程式（GRACE）
# ---------------------------

def train_grace(encoder, proj, loader, optimizer, scheduler, config, device, tag="", probe_data=None):
    encoder.train()
    if proj is not None:
        proj.train()

    best_loss = float('inf')
    patience_counter = 0
    loss_history = []

    # Probe 紀錄
    probe_history = []
    probe_epoch_index = []
    best_probe = -1.0
    best_state = None

    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = max(1, int(config.gradient_accumulation))
    
    # 檢查調度器類型
    is_plateau_scheduler = isinstance(scheduler, ReduceLROnPlateau)
    
    # 激進的記憶體管理
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    for epoch in range(config.epochs):
        updates = 0
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)
        did_optim_step = False

        for step, data in enumerate(loader, start=1):
            # 在每次迭代開始時清理記憶體
            if step % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            data = data.to(device, non_blocking=True)

            # 建立兩個 view（不丟節點），使用更保守的增廣參數
            x1, e1 = augment_view(data.x, data.edge_index, 
                                feat_mask_p=config.feat_mask * 0.8,  # 降低增廣強度
                                edge_drop_p=config.edge_drop * 0.8)
            x2, e2 = augment_view(data.x, data.edge_index, 
                                feat_mask_p=config.feat_mask * 1.2, 
                                edge_drop_p=config.edge_drop * 1.2)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                # 分別計算兩個 view，避免同時存在記憶體中
                z1 = encoder(x1, e1)
                if proj is not None:
                    z1 = proj(z1)
                
                # 清理第一個 view 的中間變數
                del x1, e1
                
                z2 = encoder(x2, e2)
                if proj is not None:
                    z2 = proj(z2)
                
                # 清理第二個 view 的中間變數  
                del x2, e2
                
                # 使用配置的 chunk 計算損失，讓函式內自己處理 clamp
                loss = nt_xent_loss_chunked(z1.float(), z2.float(), tau=config.tau,
                                            chunk=config.nce_chunk) / accum_steps

            scaler.scale(loss).backward()

            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + (list(proj.parameters()) if proj else []),
                                               max_norm=config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                did_optim_step = True

            total_loss += loss.item() * accum_steps
            num_batches += 1

            # 立即清理所有中間變數
            del data, z1, z2, loss
            
            # 更頻繁的記憶體清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(1, num_batches)
        
        if did_optim_step:
            if is_plateau_scheduler:
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        if epoch % config.print_every == 0 or epoch == config.epochs - 1:
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"Epoch {epoch:03d}/{config.epochs}  Loss: {avg_loss:.6f}  LR: {optimizer.param_groups[0]['lr']:.2e}  GPU: {mem_used:.2f}/{mem_reserved:.2f}GB")
            else:
                print(f"Epoch {epoch:03d}/{config.epochs}  Loss: {avg_loss:.6f}  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 以 loss 做 best（若不啟用 probe 早停）
        # 放寬tolerance，避免過早停止，特別是當loss還在8-9這麼高的時候
        improved = avg_loss < (best_loss - 1e-3)  # 從1e-4放寬到1e-3
        if improved:
            best_loss = avg_loss
            patience_counter = 0
            if config.save_best and not config.earlystop_by_probe:
                torch.save(encoder.state_dict(), f"best_encoder_{tag}_by_loss.pt")
                if proj is not None:
                    torch.save(proj.state_dict(), f"best_proj_{tag}_by_loss.pt")
        else:
            patience_counter += 1

        # Probe（依設定）
        if not config.disable_probe and (((epoch + 1) % config.probe_every == 0) or (epoch == config.epochs - 1)):
            # 在 probe 之前清理記憶體
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            probe_batch = probe_data.to(device) if (probe_data is not None) else next(iter(loader)).to(device)
            ckey = f"probe:{config.probe_graph or tag or 'default'}"
            auc = link_pred_probe_auc(encoder, probe_batch, device, 
                                    num_samples=min(config.probe_samples, 1000),  # 限制 probe 樣本數
                                    cache_key=ckey, fix_probe_samples=bool(config.fix_probe_samples), proj=proj)
            r10 = recall_at_k(encoder, probe_batch, device, K=10, proj=proj)
            probe_history.append(auc)
            probe_epoch_index.append(epoch)
            
            print(f"   [Probe] epoch {epoch:03d} AUC={auc:.4f}  Recall@10={r10:.4f}")

            if config.earlystop_by_probe and (auc > best_probe + 1e-3):  # 也放寬probe的tolerance
                best_probe = auc
                best_state = {
                    'encoder': {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                    'proj': ({k: v.detach().cpu().clone() for k, v in proj.state_dict().items()} if proj else None)
                }
                patience_counter = 0

        loss_history.append(avg_loss)

        if patience_counter >= config.patience:
            print(f"🛑 早停於 epoch {epoch}")
            print(f"   當前loss: {avg_loss:.6f}, 最佳loss: {best_loss:.6f}, diff: {avg_loss - best_loss:.6f}")
            print(f"   當前學習率: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"   最佳probe: {best_probe:.4f}")
            print(f"   patience計數: {patience_counter}/{config.patience}")
            print(f"   tolerance設定: 1e-3 (已放寬)")  # 更新顯示
            break
            
        # 每個 epoch 結束後清理記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"✅ 訓練完成！最終 loss: {avg_loss:.6f}")

    # 覆蓋至最佳 probe 權重
    if config.earlystop_by_probe and best_state is not None:
        encoder.load_state_dict(best_state['encoder'])
        if proj is not None and best_state['proj'] is not None:
            proj.load_state_dict(best_state['proj'])
        if config.save_best:
            torch.save(encoder.state_dict(), f"best_encoder_{tag}_by_probe.pt")
            if proj is not None:
                torch.save(proj.state_dict(), f"best_proj_{tag}_by_probe.pt")

    # 畫圖
    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()
    ax1.plot(loss_history, label='Training Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.grid(True)
    if len(probe_history) > 0:
        ax2.plot(probe_epoch_index, probe_history, linestyle='--', label='Validation Probe AUC')
        ax2.set_ylabel('AUC')
        # 修正 AUC 顯示格式，去掉科學記號偏移
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax2.yaxis.set_major_formatter(formatter)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.title(f"GRACE: Loss + Probe(AUC) {tag}")
    out_png = f"loss_probe_{tag if tag else 'all'}.png"
    plt.tight_layout(); plt.savefig(out_png); plt.close()
    print(f"📈 曲線已儲存: {out_png}")

    gc.collect();
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return encoder, proj


# ---------------------------
# 調度器建立函數
# ---------------------------

def create_scheduler(optimizer, config):
    """根據配置建立學習率調度器"""
    if config.scheduler_type == 'plateau':
        # 基於loss的自適應調度器
        return ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=config.scheduler_factor, 
            patience=config.scheduler_patience,
            verbose=True,
            threshold=1e-4,
            cooldown=2
        )
    elif config.scheduler_type == 'step':
        # 固定步長調度器 
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=config.scheduler_patience*2, gamma=config.scheduler_factor)
    else:  # 'cosine' 或其他
        # 原始的warmup + cosine調度器
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs - config.warmup_epochs),
                                eta_min=config.lr * 0.01)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs])


# ---------------------------
# 隨機種子
# ---------------------------

def set_seed(seed: int):
    import numpy as np, random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------------
# 逐圖訓練
# ---------------------------

def train_separately(benchmarks, config, device, root_dir, global_cell_types, heads_schedule):
    print("🧩 啟用逐圖訓練模式...")
    encoder = None; proj = None; input_dim = None
    
    # 預先載入 probe 資料（如果指定）
    probe_data_global = None
    if config.probe_graph and config.probe_graph.strip():
        try:
            probe_case_path = os.path.join(root_dir, config.probe_graph.strip())
            probe_data_global = build_graph_from_case(probe_case_path, verilog_root=root_dir, global_cell_types=global_cell_types)
            probe_data_global = ensure_cpu_data(probe_data_global)
            print(f"🔍 已載入 probe 資料：{config.probe_graph} ({probe_data_global.x.shape[0]} 節點)")
        except Exception as e:
            print(f"⚠️ 無法載入 probe 資料 {config.probe_graph}: {e}")
            probe_data_global = None

    for name in benchmarks:
        print(f"\n🚀 開始訓練案例：{name}")
        # case_path = os.path.join(root_dir, name, 'bookshelf_run', 'output', name)
        case_path = os.path.join(root_dir, name)
        data = build_graph_from_case(case_path, verilog_root=root_dir, global_cell_types=global_cell_types)
        data = ensure_cpu_data(data)
        loader = DataLoader([data], batch_size=1)

        if encoder is None:
            input_dim = data.num_node_features
            try:
                from graph_builder import get_or_create_cell_id_mapping
                num_cells = len(get_or_create_cell_id_mapping())
            except Exception:
                num_cells = 854

            encoder = ConfigurableGATEncoder(
                in_channels=input_dim,
                out_channels=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                cell_embedding_dim=config.cell_embed_dim,
                num_cells=num_cells,
                heads_schedule=heads_schedule
            ).to(device)
            proj = (ProjectionHead(config.hidden_dim, config.proj_dim).to(device)) if config.use_proj else None

            params = list(encoder.parameters()) + (list(proj.parameters()) if proj else [])
            optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
            scheduler = create_scheduler(optimizer, config)

        # 使用全域 probe 資料（每個案例都用相同的 probe）
        probe_data = probe_data_global
        encoder, proj = train_grace(encoder, proj, loader, optimizer, scheduler, config, device, tag=name, probe_data=probe_data)

    torch.save(encoder.state_dict(), config.output_name)
    if proj is not None:
        torch.save(proj.state_dict(), config.output_name.replace('.pt','_proj.pt'))
    print(f"✅ 已保存最終模型至: {config.output_name}")
    return encoder, proj, input_dim


# ---------------------------
# 一次讀入所有圖訓練
# ---------------------------

def train_all_together(benchmarks, config, device, root_dir, global_cell_types, heads_schedule):
    print("🧩 一次讀入所有圖訓練...")
    graphs = []
    probe_data = None

    for name in benchmarks:
        # case_path = os.path.join(root_dir, name, 'bookshelf_run', 'output', name)
        case_path = os.path.join(root_dir, name)
        data = build_graph_from_case(case_path, verilog_root=root_dir, global_cell_types=global_cell_types)
        data = ensure_cpu_data(data)

        if config.probe_graph and config.probe_graph.strip()==name:
            probe_data = data
            print(f"   🔍 {name}: {data.x.shape[0]} 節點, {data.edge_index.shape[1]} 邊, {data.x.shape[1]} 特徵 (作為 probe)")
        else:
            print(f"   ✅ {name}: {data.x.shape[0]} 節點, {data.edge_index.shape[1]} 邊, {data.x.shape[1]} 特徵")
            graphs.append(data)

    if not graphs:
        raise ValueError('❌ 沒有圖可訓練，請檢查 --benchmarks 參數')

    loader = DataLoader(graphs, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    input_dim = graphs[0].num_node_features

    try:
        from graph_builder import get_or_create_cell_id_mapping
        num_cells = len(get_or_create_cell_id_mapping())
    except Exception:
        num_cells = 854

    # heads 解析
    heads_schedule = heads_schedule or []

    encoder = ConfigurableGATEncoder(
        in_channels=input_dim,
        out_channels=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        cell_embedding_dim=config.cell_embed_dim,
        num_cells=num_cells,
        heads_schedule=heads_schedule
    ).to(device)
    proj = (ProjectionHead(config.hidden_dim, config.proj_dim).to(device)) if config.use_proj else None

    params = list(encoder.parameters()) + (list(proj.parameters()) if proj else [])
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = create_scheduler(optimizer, config)

    encoder, proj = train_grace(encoder, proj, loader, optimizer, scheduler, config, device, tag='all', probe_data=probe_data)

    torch.save(encoder.state_dict(), config.output_name)
    if proj is not None:
        torch.save(proj.state_dict(), config.output_name.replace('.pt','_proj.pt'))
    print(f"✅ 模型已保存至: {config.output_name}")

    return encoder, proj, input_dim


# ---------------------------
# Main
# ---------------------------

def main():
    start_time = time.time()
    config = parse_args()
    set_seed(config.seed)

    # 裝置
    if config.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif config.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('🖥️ 使用裝置：', device)
    
    # 輸出 probe 狀態
    if config.disable_probe:
        print('🚫 Probe 驗證已禁用（純訓練模式）')
    else:
        print(f'🔍 Probe 驗證已啟用（每 {config.probe_every} epochs，樣本數：{config.probe_samples}）')

    # heads 解析
    heads_schedule = []
    if config.heads_schedule.strip():
        try:
            heads_schedule = [int(x) for x in config.heads_schedule.strip().split(',') if x.strip()]
        except Exception:
            heads_schedule = []

    # 自動調參（針對低顯存優化）
    gpu_mem = 16  # 預設值，避免未定義錯誤
    if device.type == 'cuda' and not config.disable_auto_tune:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'🔍 GPU 總記憶體: {gpu_mem:.1f} GB')
        if gpu_mem < 12:
            # 針對低顯存環境，更激進的參數調整
            orig_b, orig_chunk, orig_hidden = config.batch_size, config.nce_chunk, config.hidden_dim
            config.batch_size = 1  # 強制設為 1
            config.nce_chunk = 512  # 極端小 chunk (改為 512)
            # config.gradient_accumulation = max(config.gradient_accumulation, 64)  # 大幅增加梯度累積 (從 16 改為 64)
            config.cell_embed_dim = min(config.cell_embed_dim, 8)  # 極小的 embedding (改為 8)
            config.hidden_dim = min(config.hidden_dim, 64)  # 大幅降低隱藏維度 (改為 64)
            config.proj_dim = min(config.proj_dim, config.hidden_dim)  # 投影維度跟隨隱藏維度
            print('⚠️ 低顯存激進調整:')
            print(f'   batch_size: {orig_b} → {config.batch_size}')
            print(f'   nce_chunk:  {orig_chunk} → {config.nce_chunk}')
            print(f'   hidden_dim: {orig_hidden} → {config.hidden_dim}')
            print(f'   proj_dim: → {config.proj_dim}')
            print(f'   gradient_accumulation: {config.gradient_accumulation}')
            print(f'   cell_embed_dim: {config.cell_embed_dim}')
    elif device.type == 'cuda':
        # 即使禁用自動調整，也要取得 GPU 記憶體資訊
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'🔍 GPU 總記憶體: {gpu_mem:.1f} GB (自動調整已禁用)')
    
    # 啟用投影頭提升性能（但在低顯存時使用較小維度）
    if not hasattr(config, 'use_proj') or not config.use_proj:
        config.use_proj = True
        if gpu_mem < 12:
            config.proj_dim = min(config.proj_dim, config.hidden_dim)  # 投影維度不超過隱藏維度
        print(f'🚀 自動啟用 projection head (dim={config.proj_dim}) 以提升對比學習效果')

    if config.test_only:
        config.benchmarks = 's27'
        config.epochs = min(config.epochs, 10)
        print('🧪 測試模式：s27，縮短 epochs')
    else:
        # full = (
        #     'c17, c432, c499, c880, c1355, c1908, c2670, c3540, c5315, c6288, '
        #     's27, s298, s344, s382, s386, s400, s1196, s1238, s1423, s1488, s5378, s9234, s13207, s15850, s38584'
        # )
        full = (
            # 'c17, c432, c499, c880, c1355, c1908, c2670, c3540, c5315, c6288, '
            # 's27, s298, s344, s382, s386, s400, s1196, s1238, s1423, s1488, s5378, s9234, s13207, s15850, s38584'
            # 'des, aes, ac97_top, aes_cipher_top, pci_bridge32, ariane'
            'des, aes, ac97_top, aes_cipher_top'
            # 'des'
        )
        config.benchmarks = full

    benchmarks = [b.strip() for b in config.benchmarks.split(',') if b.strip()]

    # 根目錄（依你環境）
    root_dir = '/root/solution/testcases'

    # 收集全域 cell types（供 cell embedding）
    global_cell_types = collect_all_cell_types(benchmarks, root_dir, root_dir)

    if config.separate_training:
        encoder, proj, input_dim = train_separately(benchmarks, config, device, root_dir, global_cell_types, heads_schedule)
    else:
        encoder, proj, input_dim = train_all_together(benchmarks, config, device, root_dir, global_cell_types, heads_schedule)

    # 訓練後：如需多次隨機抽樣平均（泛化報告）
    if not config.disable_probe and config.final_probe_average > 0 and config.probe_graph:
        # case_path = os.path.join(root_dir, config.probe_graph, 'bookshelf_run', 'output', config.probe_graph)
        case_path = os.path.join(root_dir, config.probe_graph)
        probe_data = build_graph_from_case(case_path, verilog_root=root_dir, global_cell_types=global_cell_types)
        probe_data = ensure_cpu_data(probe_data)
        device_eval = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vals = []
        for _ in range(config.final_probe_average):
            auc = link_pred_probe_auc(encoder, probe_data, device_eval, num_samples=config.probe_samples,
                                      cache_key=None, fix_probe_samples=False, proj=proj)
            vals.append(auc)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        std = math.sqrt(var)
        print(f"📊 Final Probe AUC over {config.final_probe_average} runs: mean={mean:.4f}, std={std:.4f}, vals={[round(v,4) for v in vals]}")

    # 輸出 meta
    if config.save_meta:
        meta = {
            'method': 'GRACE',
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'dropout': config.dropout,
            'epochs': config.epochs,
            'learning_rate': config.lr,
            'weight_decay': config.weight_decay,
            'separate_training': config.separate_training,
            'feature_dim': int(input_dim),
            'model_file': config.output_name,
            'seed': config.seed,
            'benchmarks': benchmarks,
            'probe_every': config.probe_every,
            'probe_samples': config.probe_samples,
            'earlystop_by_probe': bool(config.earlystop_by_probe),
            'probe_graph': config.probe_graph,
            'fix_probe_samples': bool(config.fix_probe_samples),
            'final_probe_average': int(config.final_probe_average),
            'heads_schedule': heads_schedule if heads_schedule else 'default',
            'tau': config.tau,
            'feat_mask': config.feat_mask,
            'edge_drop': config.edge_drop,
            'use_proj': bool(config.use_proj),
            'proj_dim': config.proj_dim,
            'cell_embed_dim': config.cell_embed_dim,  # 新增：cell embedding 維度
            'batch_size': config.batch_size,           # 新增：批次大小
            'gradient_accumulation': config.gradient_accumulation,  # 新增：梯度累積
            'nce_chunk': config.nce_chunk,            # 新增：NCE 分塊大小
            'scheduler_type': config.scheduler_type,  # 新增：調度器類型
            'scheduler_patience': config.scheduler_patience,  # 新增：調度器耐心
            'scheduler_factor': config.scheduler_factor,      # 新增：調度器因子
        }
        with open('encoder_meta.json', 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print('📝 已輸出模型中繼資料: encoder_meta.json')

    elapsed = time.time() - start_time
    print(f"⏱️ 訓練完成！總耗時: {elapsed:.2f} 秒")
    print(f"📊 最終節點 embedding 維度: {config.proj_dim if config.use_proj else config.hidden_dim}")


if __name__ == '__main__':
    main()
