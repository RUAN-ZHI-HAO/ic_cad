# -*- coding: utf-8 -*-
"""
DGI 訓練（含 Validation Probe）
- 每隔 N 個 epoch 以「連結預測 AUC」做驗證探頭
- 圖出 Loss + Probe(AUC) 雙曲線
- 可選用 Probe(AUC) 作為早停與最佳模型保存準則
- 支援固定抽樣 (穩定早停/監控) 與訓練結束多次隨機平均 (泛化報告)
"""

import argparse
import os
import sys
import json
import time
import gc
import random
import datetime
import math

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.nn import GATConv
from torch_geometric.nn.models import DeepGraphInfomax
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling, subgraph, k_hop_subgraph

from graph_builder import build_graph_from_case, collect_all_cell_types

# 固定驗證抽樣的全域快取：key -> (pos_edge_index, neg_edge_index)
PROBE_CACHE = {}


def ensure_cpu_data(data):
    """保證 PyG Data 全部張量都在 CPU（避免 DataLoader collation 出現 device 混用）"""
    return data.to('cpu')


# ---------------------------
# 參數
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='可配置的 DGI 訓練（含 Validation Probe）')

    # 模型架構參數
    parser.add_argument('--hidden-dim', type=int, default=32, help='隱藏層維度 (預設: 32)')
    parser.add_argument('--num-layers', type=int, default=2, help='GAT 層數 (預設: 2)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 比率 (預設: 0.1)')
    parser.add_argument('--cell-embed-dim', type=int, default=16, help='Cell embedding 維度 (預設: 16)')
    parser.add_argument('--heads-schedule', type=str, default='', help='每層注意力頭數，如 "2,1,1"；空字串則使用預設')

    # 訓練參數
    parser.add_argument('--epochs', type=int, default=200, help='訓練輪數 (預設: 200)')
    parser.add_argument('--lr', type=float, default=3e-4, help='學習率 (預設: 3e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='權重衰減 (預設: 1e-5)')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小 (預設: 16)')
    parser.add_argument('--gradient-accumulation', type=int, default=1, help='梯度累積步數，模擬更大 batch (預設: 1=不累積)')

    # 記憶體/大圖處理參數
    parser.add_argument('--edge-sampling', action='store_true', help='對超大圖進行邊分級採樣以省顯存')
    parser.add_argument('--edge-sampling-large', type=float, default=0.5, help='>50萬邊圖的保留比例 (預設: 0.5)')
    parser.add_argument('--edge-sampling-huge', type=float, default=0.1, help='>100萬邊圖的保留比例 (預設: 0.1)')

    # 早停和調度參數
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值 (預設: 20)')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪 (預設: 1.0)')
    parser.add_argument('--scheduler-factor', type=float, default=0.5, help='學習率衰減因子 (預設: 0.5)')
    parser.add_argument('--scheduler-patience', type=int, default=10, help='學習率調度耐心值 (預設: 10)')

    # 輸出控制
    parser.add_argument('--print-every', type=int, default=10, help='輸出頻率 (預設: 每10輪)')
    parser.add_argument('--save-best', action='store_true', help='保存最佳模型')
    parser.add_argument('--output-name', type=str, default="encoder.pt", help='輸出模型檔名 (預設: encoder.pt)')

    # 數據設定
    parser.add_argument('--benchmarks', type=str, default="c17,c432,c499,c880,c1355", help='訓練電路 (逗號分隔)')
    parser.add_argument('--test-only', action='store_true', help='僅測試模式（使用 c17）')
    parser.add_argument('--separate-training', action='store_true', help='逐圖訓練模式')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子 (預設: 42)')
    parser.add_argument('--save-meta', action='store_true', help='輸出 meta.json 方便推論載入')

    # Validation Probe 參數
    parser.add_argument('--probe-every', type=int, default=10, help='每幾個 epoch 做一次驗證探頭 (預設: 10)')
    parser.add_argument('--probe-samples', type=int, default=3000, help='每次探頭取多少正/負邊 (預設: 3000)')
    parser.add_argument('--earlystop-by-probe', action='store_true', help='用 probe(AUC) 最佳點早停/選模')
    parser.add_argument('--probe-graph', type=str, default="", help='指定用哪個基準圖做 probe（空字串則用 loader 第一個 batch）')
    parser.add_argument('--fix-probe-samples', action='store_true', help='訓練期間固定同一批 probe 正/負邊')
    parser.add_argument('--final-probe-average', type=int, default=0, help='訓練結束後再做 N 次隨機抽樣，回報平均AUC±std (0=關閉)')

    # 裝置參數
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='運算裝置：auto/cuda/cpu (預設: auto)')
    parser.add_argument('--disable-auto-tune', action='store_true',help='關閉自動調整 batch/probe 參數（強制使用使用者指定值）')

    return parser.parse_args()


# ---------------------------
# DGI 所需：corruption（負樣本）
# ---------------------------
def corruption(x, edge_index, drop_edge_p=0.1, feat_mask_p=0.1):
    # 1) feature shuffle
    perm = torch.randperm(x.size(0), device=x.device)
    x_corr = x[perm].clone()

    # 2) random feature mask（不改最後一維 cell_id）
    if feat_mask_p > 0:
        mask = torch.rand_like(x_corr[:, :-1]) < feat_mask_p
        x_corr[:, :-1][mask] = 0

    # 3) random edge drop
    if drop_edge_p > 0 and edge_index.numel() > 0:
        E = edge_index.size(1)
        keep = torch.rand(E, device=edge_index.device) > drop_edge_p
        edge_index = edge_index[:, keep]

    return x_corr, edge_index


# ---------------------------
# GAT Encoder
# ---------------------------
class ConfigurableGATEncoder(torch.nn.Module):
    """可配置的 GAT 編碼器（支持 cell embedding），默認 concat=False 降顯存"""
    def __init__(self, in_channels, out_channels, num_layers=2, heads_schedule=None, dropout=0.1,
                 cell_embedding_dim=16, num_cells=854):
        super().__init__()

        self.cell_embedding = torch.nn.Embedding(num_cells, cell_embedding_dim)
        actual_in_channels = in_channels - 1 + cell_embedding_dim  # -1 移除 cell_id

        # 頭數排程
        if heads_schedule is None or len(heads_schedule) == 0:
            if num_layers == 1:
                heads_schedule = [1]
            elif num_layers == 2:
                heads_schedule = [2, 1]
            else:
                heads_schedule = [2] * (num_layers - 1) + [1]

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        in_dim = actual_in_channels
        for i in range(num_layers):
            heads = heads_schedule[i] if i < len(heads_schedule) else 1
            out_dim = out_channels
            self.convs.append(GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout))
            if i < num_layers - 1:
                self.bns.append(torch.nn.BatchNorm1d(out_dim))
            in_dim = out_dim
        self.dropout = dropout

    def forward(self, x, edge_index):
        base_features = x[:, :-1]
        cell_ids = x[:, -1].long()
        cell_embeds = self.cell_embedding(cell_ids)
        x = torch.cat([base_features, cell_embeds], dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ---------------------------
# Validation Probe: Link Prediction AUC（支援固定抽樣）
# ---------------------------
@torch.no_grad()
def link_pred_probe_auc(encoder, data, device, num_samples=2000, cache_key=None, fix_probe_samples=False):
    """
    - 固定抽樣：第一次抽樣後快取 pos/neg，之後重用，AUC 變動更穩。
    - 只在打分階段做分批，encoder 前向一次完整圖（保持結構）。
    """
    encoder.eval()
    x, edge_index = data.x.to(device), data.edge_index.to(device)

    use_amp = (device.type == 'cuda')
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
        z = encoder(x, edge_index)
    z = F.normalize(z, p=2, dim=-1)

    # ---- 取得/建立抽樣 ----
    pos = neg = None
    if fix_probe_samples and (cache_key is not None) and (cache_key in PROBE_CACHE):
        pos, neg = PROBE_CACHE[cache_key]

    if (pos is None) or (neg is None):
        E = edge_index.size(1)
        take = min(num_samples, E)
        idx = torch.randperm(E, device=device)[:take]
        pos = edge_index[:, idx]
        neg = negative_sampling(edge_index=edge_index, num_nodes=x.size(0), num_neg_samples=take)
        if fix_probe_samples and (cache_key is not None):
            PROBE_CACHE[cache_key] = (pos, neg)

    # ---- 分批打分避免顯存尖峰 ----
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


def final_probe_average(encoder, data, device, samples, times: int):
    """訓練結束後：重複隨機抽樣多次，回報平均與標準差"""
    vals = []
    for _ in range(times):
        auc = link_pred_probe_auc(
            encoder, data, device,
            num_samples=samples,
            cache_key=None,
            fix_probe_samples=False
        )
        vals.append(auc)
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    std = math.sqrt(var)
    return mean, std, vals


# ---------------------------
# 訓練主流程（AMP + 梯度累積 + Probe）
# ---------------------------
def train_dgi_configurable(model, loader, optimizer, scheduler, config, device, tag="", probe_data=None):
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    loss_history = []
    delta = 1e-4  # 早停容忍度

    # Probe 紀錄
    probe_history = []
    probe_epoch_index = []
    best_probe = -1.0
    best_state_dict = None

    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = max(1, int(config.gradient_accumulation))
    global_step = 0

    for epoch in range(config.epochs):
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for step, data in enumerate(loader, start=1):
            data = data.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                pos_z, neg_z, summary = model(data.x, data.edge_index)
                loss = model.loss(pos_z, neg_z, summary) / accum_steps

            scaler.scale(loss).backward()

            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            total_loss += loss.item() * accum_steps
            num_batches += 1

            del data, pos_z, neg_z, summary, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(1, num_batches)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % config.print_every == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:03d}/{config.epochs}  Loss: {avg_loss:.6f}  LR: {current_lr:.2e}")

        # Best (by loss)
        if avg_loss < best_loss - delta:
            best_loss = avg_loss
            patience_counter = 0
            if config.save_best and not config.earlystop_by_probe:
                torch.save(model.encoder.state_dict(), f"best_encoder_{tag}_by_loss.pt")
        else:
            patience_counter += 1

        # ---- Validation Probe（依設定每 N 週期觸發）----
        if ((epoch + 1) % config.probe_every == 0) or (epoch == config.epochs - 1):
            if probe_data is None:
                try:
                    probe_batch = next(iter(loader))
                except StopIteration:
                    probe_batch = None
            else:
                probe_batch = probe_data

            if probe_batch is not None:
                probe_batch = probe_batch.to(device)
                # 決定固定抽樣鍵
                if config.probe_graph:
                    ckey = f"probe:{config.probe_graph}"
                elif tag:
                    ckey = f"probe:{tag}"
                else:
                    ckey = "probe:default"

                auc = link_pred_probe_auc(
                    model.encoder,
                    probe_batch,
                    device,
                    num_samples=config.probe_samples,
                    cache_key=ckey,
                    fix_probe_samples=bool(config.fix_probe_samples)
                )
                probe_history.append(auc)
                probe_epoch_index.append(epoch)
                if epoch % config.print_every == 0 or epoch == config.epochs - 1:
                    print(f"   [Probe] epoch {epoch:03d} AUC={auc:.4f}")

                # 以 Probe 為選模/早停準則
                if config.earlystop_by_probe and (auc > best_probe + 1e-4):
                    best_probe = auc
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
                    patience_counter = 0  # 以 probe 改寫耐心

        loss_history.append(avg_loss)

        # 早停
        if patience_counter >= config.patience:
            print(f"🛑 早停於 epoch {epoch}（best_loss={best_loss:.6f}, best_probe={best_probe:.4f}）")
            break

    print(f"✅ 訓練完成！最終 loss: {avg_loss:.6f}")

    # 若以 probe 為準則，覆蓋 encoder 為最佳 probe 權重
    if config.earlystop_by_probe and best_state_dict is not None:
        model.encoder.load_state_dict(best_state_dict)
        if config.save_best:
            torch.save(model.encoder.state_dict(), f"best_encoder_{tag}_by_probe.pt")

    # 繪圖：Loss + Probe(AUC)
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(loss_history, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    if len(probe_history) > 0:
        ax2.plot(probe_epoch_index, probe_history, linestyle="--", label="Validation Probe AUC")
        ax2.set_ylabel("AUC")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"DGI: Loss + Probe(AUC) {tag}")
    out_png = f"loss_probe_{tag if tag else 'all'}.png"
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"📈 曲線已儲存: {out_png}")
    plt.close()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


# ---------------------------
# 設定隨機種子
# ---------------------------
def set_seed(seed: int):
    import numpy as np, random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN 設定：關閉嚴格模式，保留速度與基本穩定性
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------------
# 逐圖訓練
# ---------------------------
def train_separately(benchmarks, config, device, root_dir, global_cell_types, heads_schedule):
    print("🧩 啟用逐圖訓練模式...")
    model = None
    input_dim = None

    for name in benchmarks:
        case_path = os.path.join(root_dir, name, "bookshelf_run", "output", name)
        data = build_graph_from_case(case_path, verilog_root=root_dir, global_cell_types=global_cell_types)
        data = ensure_cpu_data(data)
        loader = DataLoader([data], batch_size=1)

        if model is None:
            input_dim = data.num_node_features
            try:
                from graph_builder import get_or_create_cell_id_mapping
                cell_to_id = get_or_create_cell_id_mapping()
                num_cells = len(cell_to_id)
            except:
                num_cells = 854

            encoder = ConfigurableGATEncoder(
                in_channels=input_dim,
                out_channels=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                cell_embedding_dim=config.cell_embed_dim,
                num_cells=num_cells,
                heads_schedule=heads_schedule
            )
            model = DeepGraphInfomax(
                hidden_channels=config.hidden_dim,
                encoder=encoder,
                summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                corruption=corruption,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience
            )

        probe_data = None
        if config.probe_graph and config.probe_graph.strip() == name:
            probe_data = data

        model = train_dgi_configurable(
            model, loader, optimizer, scheduler, config, device, tag=name, probe_data=probe_data
        )

    torch.save(model.encoder.state_dict(), config.output_name)
    print(f"✅ 已保存最終模型至: {config.output_name}")
    return model, input_dim


# ---------------------------
# 一次讀入所有圖訓練（含大圖分級採樣選項）
# ---------------------------
def train_all_together(benchmarks, config, device, root_dir, global_cell_types, heads_schedule):
    print("🧩 一次讀入所有圖訓練...")
    graph_list = []
    probe_data = None

    for name in benchmarks:
        case_path = os.path.join(root_dir, name, "bookshelf_run", "output", name)
        data = build_graph_from_case(case_path, verilog_root=root_dir, global_cell_types=global_cell_types)
        data = ensure_cpu_data(data)

        if config.probe_graph and config.probe_graph.strip() == name:
            probe_data = data
            print(f"   🔍 {name}: {data.x.shape[0]} 節點, {data.edge_index.shape[1]} 邊, {data.x.shape[1]} 特徵 (作為 probe，不參與訓練)")
        else:
            # 可選：對大圖做邊分級採樣
            if config.edge_sampling:
                num_edges = data.edge_index.shape[1]
                before = num_edges
                ratio = None
                if num_edges > 1_000_000:
                    ratio = float(config.edge_sampling_huge)
                elif num_edges > 500_000:
                    ratio = float(config.edge_sampling_large)

                # if ratio is not None and 0 < ratio < 1.0:
                #     perm = torch.randperm(num_edges)[:int(num_edges * ratio)]
                #     data.edge_index = data.edge_index[:, perm]
                #     after = data.edge_index.shape[1]
                #     print(f"   🔧 {name}: {data.x.shape[0]} 節點, {before}→{after} 邊 (採樣 {int(ratio*100)}%), {data.x.shape[1]} 特徵")
                # else:
                print(f"   ✅ {name}: {data.x.shape[0]} 節點, {num_edges} 邊, {data.x.shape[1]} 特徵")
            else:
                print(f"   ✅ {name}: {data.x.shape[0]} 節點, {data.edge_index.shape[1]} 邊, {data.x.shape[1]} 特徵")

            graph_list.append(data)

    if not graph_list:
        raise ValueError("❌ 沒有圖片可以訓練！請檢查 --benchmarks 或關閉過度嚴格的過濾/採樣")

    loader = DataLoader(graph_list, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    input_dim = graph_list[0].num_node_features

    try:
        from graph_builder import get_or_create_cell_id_mapping
        cell_to_id = get_or_create_cell_id_mapping()
        num_cells = len(cell_to_id)
    except:
        num_cells = 854

    encoder = ConfigurableGATEncoder(
        in_channels=input_dim,
        out_channels=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        cell_embedding_dim=config.cell_embed_dim,
        num_cells=num_cells,
        heads_schedule=heads_schedule
    )
    model = DeepGraphInfomax(
        hidden_channels=config.hidden_dim,
        encoder=encoder,
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience
    )

    trained_model = train_dgi_configurable(
        model, loader, optimizer, scheduler, config, device, tag="all", probe_data=probe_data
    )
    torch.save(trained_model.encoder.state_dict(), config.output_name)
    print(f"✅ 模型已保存至: {config.output_name}")
    return trained_model, input_dim


# ---------------------------
# Main
# ---------------------------
def main():
    start_time = time.time()
    config = parse_args()
    set_seed(config.seed)

    # 裝置選擇
    if config.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif config.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('🖥️ 使用裝置：', device)

    # heads_schedule 解析
    heads_schedule = []
    if config.heads_schedule.strip():
        try:
            heads_schedule = [int(x) for x in config.heads_schedule.strip().split(',') if x.strip()]
        except Exception:
            heads_schedule = []

    # 顯存檢查與自動調整
    if device.type == 'cuda' and not config.disable_auto_tune:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f'🔍 GPU 總記憶體: {gpu_mem:.1f} GB')
        if gpu_mem < 12:
            original_batch = config.batch_size
            original_probe = config.probe_samples
            config.batch_size = min(config.batch_size, 2)
            config.probe_samples = min(config.probe_samples, 300)
            print(f'⚠️  GPU記憶體不足12GB，自動調整參數：')
            print(f'   batch_size: {original_batch} → {config.batch_size}')
            print(f'   probe_samples: {original_probe} → {config.probe_samples}')

    if config.test_only:
        config.benchmarks = "c17"
        config.epochs = 10
        print("🧪 測試模式：僅使用 c17，10 個 epochs")
    else:
        fullBenchmarks = (
            "c17, c432, c499, c880, c1355, c1908, c2670, c3540, c5315, c6288, "
            "s27, s298, s344, s382, s386, s400, s1196, s1238, s1423, s1488, s5378, s9234, s13207, s15850, s38584"
        )
        config.benchmarks = fullBenchmarks

    benchmarks = [b.strip() for b in config.benchmarks.split(",") if b.strip()]

    # 根目錄（依你的環境）
    root_dir = "/root/ruan_workspace/gtlvl_design"

    # 收集全域 cell types（為了 cell embedding 的 num_cells）
    global_cell_types = collect_all_cell_types(benchmarks, root_dir, root_dir)

    if config.separate_training:
        model, input_dim = train_separately(benchmarks, config, device, root_dir, global_cell_types, heads_schedule)
    else:
        model, input_dim = train_all_together(benchmarks, config, device, root_dir, global_cell_types, heads_schedule)

    # 訓練後：若設定，做多次隨機抽樣平均（泛化報告）
    if config.final_probe_average > 0 and config.probe_graph:
        case_path = os.path.join(root_dir, config.probe_graph, "bookshelf_run", "output", config.probe_graph)
        probe_data = build_graph_from_case(case_path, verilog_root=root_dir, global_cell_types=global_cell_types)
        probe_data = ensure_cpu_data(probe_data)
        device_eval = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mean, std, vals = final_probe_average(model.encoder, probe_data, device_eval, config.probe_samples, config.final_probe_average)
        print(f"📊 Final Probe AUC over {config.final_probe_average} runs: mean={mean:.4f}, std={std:.4f}, vals={[round(v,4) for v in vals]}")

    # 輸出 meta
    if config.save_meta:
        meta = {
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'benchmarks': benchmarks,
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
            'probe_every': config.probe_every,
            'probe_samples': config.probe_samples,
            'earlystop_by_probe': bool(config.earlystop_by_probe),
            'probe_graph': config.probe_graph,
            'fix_probe_samples': bool(config.fix_probe_samples),
            'final_probe_average': int(config.final_probe_average),
            'heads_schedule': heads_schedule if heads_schedule else 'default'
        }
        with open('encoder_meta.json', 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print('📝 已輸出模型中繼資料: encoder_meta.json')

    elapsed = time.time() - start_time
    print(f"⏱️ 訓練完成！總耗時: {elapsed:.2f} 秒")
    print(f"📊 最終模型輸出維度: {config.hidden_dim}（每個節點 embedding 維度）")


if __name__ == "__main__":
    main()
