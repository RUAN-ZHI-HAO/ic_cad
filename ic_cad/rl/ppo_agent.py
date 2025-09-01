# -*- coding: utf-8 -*-
"""
2D Action Space PPO Agent for IC CAD Optimization

- 真正的 PPO（ratio/clip/entropy/multi-epoch/early-KL/grad clip）
- 替換（replacement）分支加入「替換特徵」embedding（不再用 global 當 placeholder）
- logits/mask 裡的 NaN/Inf/All-masked 向量化處理（逐 batch fallback）
- replacement_logits 建立在正確的 device
- _prepare_features 的隨機 fallback 改為零向量（避免分布飄移）
- 訓練監控新增 clip_fraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from torch.distributions import Categorical
import random
import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------
# Feature Extractor
# -----------------------------
class GNNFeatureExtractor(nn.Module):
    """
    GNN-based feature extractor for circuit elements with cell embedding support
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, 
                 cell_embedding_dim: int = 32, num_cells: int = 854):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Cell embedding layer（保留接口；目前 node_features 已是 GNN 輸出，先不使用）
        self.cell_embedding = nn.Embedding(num_cells, cell_embedding_dim)

        # 投影到 hidden 維度
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 「簡化版 GNN」：多層前饋 + skip
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (B, C, F) - 已是 GNN 嵌入特徵
        Returns:
            node_embeddings: (B, C, H)
        """
        x = self.input_proj(node_features)
        x = F.relu(x)
        for layer in self.gnn_layers:
            residual = x
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual  # Skip connection
        x = self.output_proj(x)
        return x

# -----------------------------
# Policy & Value Networks
# -----------------------------
class TwoDimensionalPolicyNetwork(nn.Module):
    """
    Policy network for 2D action space (candidate selection + replacement selection)
    """
    def __init__(self, feature_dim: int, hidden_dim: int,
                 max_candidates: int = 20, max_replacements: int = 20,
                 global_feature_dim: int = 9, num_cells: int = 854):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_candidates = max_candidates
        self.max_replacements = max_replacements
        self.global_feature_dim = global_feature_dim
        
        # GNN features for candidates
        self.gnn = GNNFeatureExtractor(feature_dim, hidden_dim, num_cells=num_cells)
        
        # Global feature processing
        self.global_processor = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 「替換選項」的 embedding（最小可行，若你有真實 replacement feature bank 可替換這一層）
        self.repl_embed = nn.Embedding(max_replacements, hidden_dim)

        # Candidate head: [cand(H) + global(H)] -> 1 logit
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Replacement head: [cand(H) + repl(H) + global(H)] -> 1 logit
        self.replacement_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        # 較小的增益避免初期過激
        for module in [self.candidate_head, self.replacement_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
        
    def forward(self,
                candidate_gnn_features: torch.Tensor,      # (B, C, F)
                candidate_dynamic_features: torch.Tensor,  # (B, C, D) 目前未用，可擴充
                global_features: torch.Tensor,             # (B, G)
                action_mask: Optional[Dict] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            candidate_logits:   (B, C)
            replacement_logits: (B, C, R)
        """
        B, C, _ = candidate_gnn_features.shape
        device = candidate_gnn_features.device

        # ---- Candidate branch ----
        cand_emb = self.gnn(candidate_gnn_features)              # (B, C, H)
        cand_emb = torch.clamp(cand_emb, -10, 10)
        glob = self.global_processor(global_features)            # (B, H)
        glob = torch.clamp(glob, -10, 10)
        glob_exp = glob.unsqueeze(1).expand(-1, C, -1)           # (B, C, H)

        cand_inp = torch.cat([cand_emb, glob_exp], dim=-1)       # (B, C, 2H)
        cand_logits = self.candidate_head(cand_inp).squeeze(-1)  # (B, C)
        cand_logits = torch.clamp(cand_logits, -15, 15)

        # ---- Replacement branch ----
        R = self.max_replacements
        replacement_logits = torch.zeros(B, C, R, device=device, dtype=cand_logits.dtype)

        # 預先建立 replacement embedding 與 global 展開
        repl_ids = torch.arange(R, device=device)                # (R,)
        repl_emb = self.repl_embed(repl_ids).unsqueeze(0).expand(B, -1, -1)  # (B, R, H)
        glob_rep = glob.unsqueeze(1).expand(-1, R, -1)           # (B, R, H)

        # 逐候選計算對應的替換 logits（此處沿用簡單迴圈，因每個 cand 有不同條件）
        for ci in range(C):
            ci_emb = cand_emb[:, ci:ci+1, :].expand(-1, R, -1)   # (B, R, H)
            repl_inp = torch.cat([ci_emb, repl_emb, glob_rep], dim=-1)  # (B, R, 3H)
            repl_logits_ci = self.replacement_head(repl_inp).squeeze(-1)  # (B, R)
            replacement_logits[:, ci, :] = torch.clamp(repl_logits_ci, -15, 15)

        # ---- 套用 candidate mask（如果提供）----
        if action_mask is not None and 'candidate_mask' in action_mask:
            cmask = torch.as_tensor(action_mask['candidate_mask'],
                                    device=device, dtype=torch.bool)   # (C_mask,)
            
            # 確保 mask 長度與 candidate logits 匹配
            actual_C = cand_logits.shape[1]  # 實際的候選數量
            if cmask.numel() < actual_C:
                # 尺寸不足時，補 False
                pad = torch.zeros(actual_C - cmask.numel(), dtype=torch.bool, device=device)
                cmask = torch.cat([cmask, pad], dim=0)
            elif cmask.numel() > actual_C:
                # 尺寸過大時，截斷
                cmask = cmask[:actual_C]
            
            invalid = ~cmask
            # 對每個 batch 套用
            cand_logits[:, invalid] = float('-inf')
            # per-batch fallback：若該 batch 全為 -inf，強制第 0 個可選
            all_masked = torch.isinf(cand_logits).all(dim=1)  # (B,)
            if actual_C > 0:  # 確保有候選可選
                cand_logits[all_masked, 0] = 0.0

        return cand_logits, replacement_logits
    
    @torch.no_grad()
    def _mask_and_sample(self,
                         candidate_logits: torch.Tensor,         # (B, C)
                         replacement_logits: torch.Tensor,       # (B, C, R)
                         action_mask: Optional[Dict],
                         deterministic: bool = False
                         ) -> Tuple[Tuple[int, int], torch.Tensor, torch.Tensor]:
        """
        採樣動作並回傳 log_prob/entropy（目前預設 B=1；保留多 batch 兼容）
        """
        device = candidate_logits.device
        B, C = candidate_logits.shape
        assert B >= 1, "Expect batch >= 1"

        # ---- sample candidate ----
        cand_dist = Categorical(logits=candidate_logits)
        if deterministic:
            cand_act = torch.argmax(candidate_logits, dim=-1)    # (B,)
        else:
            cand_act = cand_dist.sample()                        # (B,)
        cand_logp = cand_dist.log_prob(cand_act)                 # (B,)
        cand_ent = cand_dist.entropy()                           # (B,)

        # ---- build replacement logits for selected candidate (per batch) ----
        # 先取第一個 batch（你目前環境是單樣本互動）
        b = 0
        ci = int(cand_act[b].item())
        ci = min(ci, replacement_logits.shape[1] - 1)

        repl_logits_sel = replacement_logits[b, ci, :].clone()   # (R,)

        # 套用 replacement mask（若提供 per-candidate）
        if action_mask is not None and 'candidate_replacement_masks' in action_mask:
            rmasks = action_mask['candidate_replacement_masks']
            if ci < len(rmasks):
                rmask_list = rmasks[ci]  # 原始 mask（可能是 list）
                
                # 關鍵修正：確保不超出實際替換選項範圍
                actual_num_replacements = len(rmask_list)
                
                # 先將所有超出實際範圍的 logits 設為 -inf
                if actual_num_replacements < repl_logits_sel.numel():
                    repl_logits_sel[actual_num_replacements:] = float('-inf')
                
                # 轉換為 tensor 並應用 mask
                rmask = torch.as_tensor(rmask_list, device=device, dtype=torch.bool)
                
                # 確保 mask 長度不超過 repl_logits_sel
                if rmask.numel() > repl_logits_sel.numel():
                    rmask = rmask[:repl_logits_sel.numel()]
                
                # 應用 mask 到對應的 logits
                invalid = ~rmask
                repl_logits_sel[:rmask.numel()][invalid] = float('-inf')
                
                # 安全檢查：確保至少有一個有效選項
                if torch.isinf(repl_logits_sel).all():
                    # 找第一個有效選項
                    valid_indices = torch.where(rmask)[0]
                    if len(valid_indices) > 0:
                        first_valid = valid_indices[0].item()
                        if first_valid < repl_logits_sel.numel():
                            repl_logits_sel[first_valid] = 0.0
                    else:
                        # 如果沒有有效選項，強制第0個（作為最後安全措施）
                        if repl_logits_sel.numel() > 0:
                            repl_logits_sel[0] = 0.0

        # ---- sample replacement ----
        repl_dist = Categorical(logits=repl_logits_sel)
        if deterministic:
            repl_act = torch.argmax(repl_logits_sel)
        else:
            repl_act = repl_dist.sample()
        
        repl_logp = repl_dist.log_prob(repl_act)                 # ()
        repl_ent = repl_dist.entropy()                           # ()

        total_logp = cand_logp[b] + repl_logp
        total_ent = cand_ent[b] + repl_ent

        action = (int(cand_act[b].item()), int(repl_act.item()))
        return action, total_logp, total_ent

    def get_action_and_log_prob(self,
                                candidate_gnn_features: torch.Tensor,
                                candidate_dynamic_features: torch.Tensor,
                                global_features: torch.Tensor,
                                action_mask: Optional[Dict] = None,
                                deterministic: bool = False
                                ) -> Tuple[Tuple[int, int], torch.Tensor, torch.Tensor]:
        cand_logits, repl_logits = self.forward(
            candidate_gnn_features, candidate_dynamic_features, global_features, action_mask
        )
        # NaN/Inf 防護
        cand_logits = torch.where(torch.isfinite(cand_logits), cand_logits, torch.full_like(cand_logits, -10.0))
        repl_logits = torch.where(torch.isfinite(repl_logits), repl_logits, torch.full_like(repl_logits, -10.0))
        return self._mask_and_sample(cand_logits, repl_logits, action_mask, deterministic)

class TwoDimensionalValueNetwork(nn.Module):
    """Value network for state value estimation"""
    def __init__(self, feature_dim: int, hidden_dim: int,
                 global_feature_dim: int = 9, num_cells: int = 854):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.global_feature_dim = global_feature_dim
        
        self.gnn = GNNFeatureExtractor(feature_dim, hidden_dim, num_cells=num_cells)
        self.global_processor = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()
        
    def _init_weights(self):
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
        
    def forward(self,
                candidate_gnn_features: torch.Tensor,   # (B, C, F)
                global_features: torch.Tensor           # (B, G)
                ) -> torch.Tensor:
        node_embeddings = self.gnn(candidate_gnn_features)        # (B, C, H)
        graph_embedding = torch.mean(node_embeddings, dim=1)      # (B, H)
        processed_global = self.global_processor(global_features) # (B, H)
        combined = torch.cat([graph_embedding, processed_global], dim=-1)
        value = self.value_head(combined)                         # (B, 1)
        return value

# -----------------------------
# PPO Agent
# -----------------------------
class TwoDimensionalPPOAgent:
    """
    PPO Agent with 2D action space support
    """
    def __init__(self, feature_dim: int = 23, hidden_dim: int = 128, 
                 max_candidates: int = 20, max_replacements: int = 20,
                 lr: float = 3e-4, critic_lr: Optional[float] = 1e-3,
                 gamma: float = 0.99, eps_clip: float = 0.2,
                 ppo_epochs: int = 4, mini_batch_size: int = 32,
                 entropy_coef: float = 0.02, value_coef: float = 0.5,
                 target_kl: float = 0.03, max_grad_norm: float = 0.5,
                 device: str = "auto"):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_candidates = max_candidates
        self.max_replacements = max_replacements
        self.lr = lr
        self.critic_lr = critic_lr if critic_lr is not None else lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        
        # 設備配置
        import torch
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"🖥️  PPO Agent 使用設備: {self.device}")
        
        # 取得 cell 數量（可選）
        try:
            import sys
            sys.path.append('/root/ruan_workspace/ic_cad')
            from gnn.graph_builder import get_or_create_cell_id_mapping
            cell_to_id = get_or_create_cell_id_mapping()
            num_cells = len(cell_to_id)
        except Exception:
            num_cells = 854  # fallback
        
        # Networks
        self.policy_network = TwoDimensionalPolicyNetwork(
            feature_dim, hidden_dim, max_candidates, max_replacements, global_feature_dim=9, num_cells=num_cells
        ).to(self.device)  # 移動到指定設備
        self.value_network = TwoDimensionalValueNetwork(
            feature_dim, hidden_dim, global_feature_dim=9, num_cells=num_cells
        ).to(self.device)  # 移動到指定設備
        
        # Optimizers（actor/critic 可不同 LR）
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.critic_lr)
        
        # Experience buffer
        self.clear_buffer()
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'total_steps': 0
        }
        logger.info(f"🤖 PPO Agent 初始化完成:")
        logger.info(f"   � 模型: feature_dim={feature_dim}, hidden_dim={hidden_dim}")
        logger.info(f"   🎯 動作空間: max_candidates={max_candidates}, max_replacements={max_replacements}")
        logger.info(f"   🔧 PPO參數: epochs={ppo_epochs}, batch_size={mini_batch_size}, clip={eps_clip}")
        logger.info(f"   📊 學習率: actor={self.lr}, critic={self.critic_lr}")
        logger.info(f"   🎲 正則化: entropy_coef={entropy_coef}, value_coef={value_coef}")
        logger.info(f"   ⚡ 優化: target_kl={target_kl}, max_grad_norm={max_grad_norm}")
        logger.info(f"   🔢 細胞數量: {num_cells} 種細胞類型")

    # ---------- Buffer ----------
    def clear_buffer(self):
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': [],
            # 重新計算 log prob / value 所需特徵
            'candidate_gnn_features': [],
            'candidate_dynamic_features': [],
            'global_features': [],
            'action_masks': []
        }

    # ---------- Acting ----------
    def get_action(self, state, deterministic: bool = False):
        """Interact with env and store transition (單步、B=1)"""
        try:
            self.policy_network.eval()
            self.value_network.eval()
            with torch.no_grad():
                # features
                cand_gnn, cand_dyn, glob, a_mask = self._prepare_features(state)
                action, log_prob, entropy = self.policy_network.get_action_and_log_prob(
                    cand_gnn, cand_dyn, glob, a_mask, deterministic
                )
                value = self.value_network(cand_gnn, glob)

                # store
                self.buffer['states'].append(state)
                self.buffer['actions'].append(action)
                self.buffer['log_probs'].append(float(log_prob.item()))
                self.buffer['values'].append(float(value.item()))
                self.buffer['candidate_gnn_features'].append(cand_gnn.squeeze(0).detach().cpu())
                self.buffer['candidate_dynamic_features'].append(cand_dyn.squeeze(0).detach().cpu())
                self.buffer['global_features'].append(glob.squeeze(0).detach().cpu())
                self.buffer['action_masks'].append(a_mask)

                info = {
                    'log_prob': float(log_prob.item()),
                    'entropy': float(entropy.item()),
                    'value': float(value.item())
                }
                return action, info

        except Exception as e:
            logger.error(f"Error in get_action: {e}", exc_info=True)
            # fallback：挑一個有效動作
            try:
                candidates = getattr(state, 'candidate_instances', [])
                if candidates and hasattr(state, 'action_mask'):
                    cand_idx = random.randint(0, len(candidates) - 1)
                    a_mask = state.action_mask
                    if 'candidate_replacement_masks' in a_mask:
                        rmask = a_mask['candidate_replacement_masks']
                        if cand_idx < len(rmask):
                            valid_repl = [i for i, v in enumerate(rmask[cand_idx]) if v]
                            repl_idx = random.choice(valid_repl) if valid_repl else 0
                        else:
                            repl_idx = 0
                    else:
                        repl_idx = 0
                    return (cand_idx, repl_idx), {'log_prob': 0.0, 'entropy': 0.0, 'value': 0.0}
            except Exception:
                pass
            return (0, 0), {'log_prob': 0.0, 'entropy': 0.0, 'value': 0.0}

    def store_reward(self, reward: float, done: bool):
        """存儲獎勵和完成狀態，確保與其他 buffer 組件同步"""
        # 檢查 buffer 一致性
        expected_length = len(self.buffer.get('log_probs', []))
        current_reward_length = len(self.buffer.get('rewards', []))
        
        if current_reward_length >= expected_length:
            logger.warning(f"⚠️ store_reward 被調用但 rewards 長度 ({current_reward_length}) >= log_probs 長度 ({expected_length})")
            return
            
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        
        # 驗證所有組件長度一致性
        lengths = {key: len(values) for key, values in self.buffer.items() if isinstance(values, list)}
        if len(set(lengths.values())) > 1:
            logger.warning(f"⚠️ Buffer 組件長度不一致: {lengths}")

    def _prepare_features(self, state):
        """將環境的 state 轉成網路輸入（fallback 改為 0 向量，避免分布飄移）"""
        # candidates
        cand_instances = getattr(state, 'candidate_instances', [])
        if hasattr(state, 'candidate_gnn_features') and state.candidate_gnn_features is not None:
            arr = np.asarray(state.candidate_gnn_features, dtype=np.float32)  # 一次堆好
            gnn_features = torch.from_numpy(arr)  # 這行不會複製，速度快
            cand_gnn = gnn_features.unsqueeze(0) if gnn_features.ndim == 2 else gnn_features
        elif cand_instances:
            C = len(cand_instances)
            cand_gnn = torch.zeros(1, C, self.feature_dim, dtype=torch.float32)
        else:
            cand_gnn = torch.zeros(1, 1, self.feature_dim, dtype=torch.float32)

        # dynamic
        if hasattr(state, 'candidate_dynamic_features') and state.candidate_dynamic_features is not None:
            arr = np.asarray(state.candidate_dynamic_features, dtype=np.float32)
            dyn = torch.from_numpy(arr)  # 比 torch.as_tensor(list) 快很多
            cand_dyn = dyn.unsqueeze(0) if dyn.ndim == 2 else dyn
        else:
            C = cand_gnn.shape[1]
            cand_dyn = torch.zeros(1, C, 10, dtype=torch.float32)

        # global
        if hasattr(state, 'global_features') and state.global_features is not None:
            glob = torch.as_tensor(state.global_features, dtype=torch.float32).unsqueeze(0)
        else:
            glob = torch.zeros(1, 9, dtype=torch.float32)

        # mask
        if hasattr(state, 'action_mask') and state.action_mask is not None:
            a_mask = state.action_mask
        else:
            C = cand_gnn.shape[1]
            candidate_mask = [1] * C
            candidate_replacement_masks = [[1] * min(20, self.max_replacements) for _ in range(C)]
            a_mask = {'candidate_mask': candidate_mask,
                      'candidate_replacement_masks': candidate_replacement_masks}

        # 轉到同一裝置（與網路相同）
        dev = next(self.policy_network.parameters()).device
        cand_gnn = cand_gnn.to(dev)
        cand_dyn = cand_dyn.to(dev)
        glob = glob.to(dev)
        return cand_gnn, cand_dyn, glob, a_mask

    # ---------- Updating ----------
    def update(self):
        """PPO 更新（ratio + clip + entropy + multi-epoch + early KL stop）"""
        num_steps = len(self.buffer['rewards'])
        if num_steps == 0:
            logger.warning("⚠️ PPO update 被調用但 buffer 為空，跳過更新")
            return {}

        # 檢查所有必需的 buffer 組件
        required_keys = ['rewards', 'log_probs', 'values', 'dones', 
                        'candidate_gnn_features', 'candidate_dynamic_features', 
                        'global_features', 'action_masks', 'actions']
        for key in required_keys:
            if key not in self.buffer or len(self.buffer[key]) == 0:
                logger.error(f"❌ Buffer 缺少必需組件 '{key}' 或為空，跳過更新")
                return {}
            if len(self.buffer[key]) != num_steps:
                logger.error(f"❌ Buffer 組件 '{key}' 長度 {len(self.buffer[key])} 與 rewards 長度 {num_steps} 不匹配")
                return {}

        device = next(self.policy_network.parameters()).device

        rewards = torch.tensor(self.buffer['rewards'], dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(self.buffer['log_probs'], dtype=torch.float32, device=device)
        old_values = torch.tensor(self.buffer['values'], dtype=torch.float32, device=device)
        dones = torch.tensor(self.buffer['dones'], dtype=torch.float32, device=device)

        # ---- GAE ----
        returns, advantages = [], []
        gae = 0.0
        lambda_gae = 0.95
        for i in reversed(range(num_steps)):
            next_v = 0.0 if i == num_steps - 1 else old_values[i + 1]
            delta = rewards[i] + self.gamma * next_v * (1.0 - dones[i]) - old_values[i]
            gae = delta + self.gamma * lambda_gae * (1.0 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + old_values[i])
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = list(range(num_steps))
        epoch_policy_losses, epoch_value_losses, epoch_entropies = [], [], []
        approx_kls, clip_fracs = [], []

        for epoch in range(self.ppo_epochs):
            random.shuffle(indices)
            # 確保 mini_batch_size 不超過可用數據量
            effective_batch_size = min(self.mini_batch_size, num_steps)
            for start in range(0, num_steps, effective_batch_size):
                batch_idx = indices[start:start + effective_batch_size]

                batch_policy_losses = []
                batch_value_losses = []
                batch_entropies = []
                batch_new_log_probs = []
                batch_clip_flags = []

                # 逐樣本（因 C/R 可變，不易直接 batch）
                for i in batch_idx:
                    cand_feat = self.buffer['candidate_gnn_features'][i].unsqueeze(0).to(device)  # (1, C, F)
                    dyn_feat = self.buffer['candidate_dynamic_features'][i].unsqueeze(0).to(device)
                    glob_feat = self.buffer['global_features'][i].unsqueeze(0).to(device)
                    action_mask = self.buffer['action_masks'][i]
                    action = self.buffer['actions'][i]  # (cand_idx, repl_idx)

                    cand_logits, repl_logits = self.policy_network.forward(
                        cand_feat, dyn_feat, glob_feat, action_mask
                    )

                    # candidate dist
                    cand_dist = Categorical(logits=cand_logits)
                    cand_logprob_all = F.log_softmax(cand_logits, dim=-1)  # (1, C)
                    cand_logprob = cand_logprob_all[0, action[0]]
                    cand_entropy = cand_dist.entropy()[0]

                    # replacement dist
                    ci = min(action[0], repl_logits.shape[1] - 1)
                    repl_logits_sel = repl_logits[0, ci, :]

                    if action_mask is not None and 'candidate_replacement_masks' in action_mask:
                        rmask = action_mask['candidate_replacement_masks']
                        if ci < len(rmask):
                            rmask_list = rmask[ci]  # 原始 mask（list）
                            actual_num_replacements = len(rmask_list)
                            
                            # 先將超出實際範圍的設為 -inf
                            repl_logits_sel = repl_logits_sel.clone()
                            if actual_num_replacements < repl_logits_sel.numel():
                                repl_logits_sel[actual_num_replacements:] = float('-inf')
                            
                            # 應用實際的 mask
                            rmask_ci = torch.as_tensor(rmask_list, device=device, dtype=torch.bool)
                            if rmask_ci.numel() > repl_logits_sel.numel():
                                rmask_ci = rmask_ci[:repl_logits_sel.numel()]
                            
                            invalid = ~rmask_ci
                            repl_logits_sel[:rmask_ci.numel()][invalid] = float('-inf')
                            
                            if torch.isinf(repl_logits_sel).all():
                                # 安全機制：找第一個有效選項
                                valid_indices = torch.where(rmask_ci)[0]
                                if len(valid_indices) > 0:
                                    first_valid = valid_indices[0].item()
                                    if first_valid < repl_logits_sel.numel():
                                        repl_logits_sel[first_valid] = 0.0
                                else:
                                    repl_logits_sel[0] = 0.0

                    repl_dist = Categorical(logits=repl_logits_sel)
                    repl_logprob_all = F.log_softmax(repl_logits_sel, dim=-1)   # (R,)
                    ri = min(action[1], repl_logits_sel.shape[0] - 1)
                    repl_logprob = repl_logprob_all[ri]
                    repl_entropy = repl_dist.entropy()

                    new_log_prob = cand_logprob + repl_logprob
                    entropy = cand_entropy + repl_entropy

                    # Value
                    new_value = self.value_network(cand_feat, glob_feat).squeeze(-1)[0]

                    adv = advantages[i]
                    ret = returns[i]
                    old_lp = old_log_probs[i]

                    # ratio 和 clip 統計
                    ratio = torch.exp(new_log_prob - old_lp)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv
                    sample_policy_loss = -torch.min(surr1, surr2)
                    sample_value_loss = F.mse_loss(new_value, ret)
                    batch_policy_losses.append(sample_policy_loss)
                    batch_value_losses.append(sample_value_loss)
                    batch_entropies.append(entropy.detach())
                    batch_new_log_probs.append(new_log_prob.detach())

                    # 記錄 ratio 統計和是否被 clip
                    is_clipped = torch.abs(ratio - 1.0) > self.eps_clip
                    batch_clip_flags.append(is_clipped.float())

                # 聚合 mini-batch
                policy_loss = torch.stack(batch_policy_losses).mean()
                value_loss = torch.stack(batch_value_losses).mean()
                entropy_mean = torch.stack(batch_entropies).mean()
                clip_fraction = torch.stack(batch_clip_flags).mean().item()

                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                # KL (approx) : old - new
                batch_old_lp = old_log_probs[batch_idx]
                batch_new_lp_tensor = torch.stack(batch_new_log_probs)
                approx_kl = (batch_old_lp - batch_new_lp_tensor).mean().item()
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_fraction)

                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy_mean.item())

            # Early stop if KL 散度太大
            if len(approx_kls) > 0:
                recent_kl = float(np.mean(approx_kls[-max(1, len(approx_kls)//4):]))
                if recent_kl > self.target_kl * 1.5:
                    logger.info(f"⏹️ PPO 早停於 epoch {epoch+1}/{self.ppo_epochs}: "
                              f"KL散度 {recent_kl:.4f} > 目標 {self.target_kl * 1.5:.4f}")
                    break

        stats = {
            'policy_loss': float(np.mean(epoch_policy_losses)) if epoch_policy_losses else 0.0,
            'value_loss': float(np.mean(epoch_value_losses)) if epoch_value_losses else 0.0,
            'entropy': float(np.mean(epoch_entropies)) if epoch_entropies else 0.0,
            'approx_kl': float(np.mean(approx_kls)) if approx_kls else 0.0,
            'clip_fraction': float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            'mean_advantage': advantages.mean().item(),
            'steps': num_steps,
            'epochs_ran': epoch + 1
        }

        self.clear_buffer()
        return stats

    # ---------- IO ----------
    def save_model(self, filepath: str):
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, filepath)
        logger.info(f"✓ 模型已保存至 {filepath}")
    
    def load_model(self, filepath: str, strict: bool = True):
        try:
            # 根據當前設備載入模型
            if self.device.type == 'cuda':
                checkpoint = torch.load(filepath, map_location=self.device)
            else:
                checkpoint = torch.load(filepath, map_location='cpu')
                
            self.policy_network.load_state_dict(checkpoint['policy_state_dict'], strict=strict)
            self.value_network.load_state_dict(checkpoint['value_state_dict'], strict=strict)
            
            # 確保網路在正確的設備上
            self.policy_network = self.policy_network.to(self.device)
            self.value_network = self.value_network.to(self.device)
            
            if 'policy_optimizer_state_dict' in checkpoint:
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            if 'value_optimizer_state_dict' in checkpoint:
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            logger.info(f"✓ 模型載入成功 ({'strict' if strict else 'non-strict'}) - 設備: {self.device}")
            return True
        except Exception as e:
            logger.error(f"✗ 模型載入失敗：{e}")
            return False
