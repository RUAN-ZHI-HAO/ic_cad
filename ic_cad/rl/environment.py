# environment.py
# -*- coding: utf-8 -*-
"""
RL Environment（OpenROAD 版）
- reset(): 一次載入設計，讀初始報告；之後 step() 不重載
- step(): 離散動作 -> OpenROAD.apply_action -> OpenROAD.report_metrics -> reward -> 狀態更新
"""

from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# 依專案結構匯入
try:
    from config import RLConfig  # 型別提示用，執行時非必要
except Exception:
    RLConfig = Any  # fallback

from utils_openroad import (
    OpenRoadInterface,
    OptimizationAction,     # fields = action_type, target_cell, new_cell_type, position
    MetricsReport,
)
from cell_replacement_manager import CellReplacementManager

logger = logging.getLogger(__name__)

# ---- 狀態表示結構 ----
@dataclass
class CandidateCell:
    """候選cell的表示"""
    name: str
    master: str
    x: float = 0.0
    y: float = 0.0

@dataclass 
class CircuitState:
    """電路狀態表示"""
    candidate_cells: List[Any]  # CandidateCell instances or List[str]
    current_tns: float
    current_wns: float
    current_power: float
    step_count: int
    circuit_metrics: Any
    candidate_gnn_features: List[List[float]]
    initial_tns: Optional[float] = None
    initial_wns: Optional[float] = None
    initial_power: Optional[float] = None

logger = logging.getLogger(__name__)

# ---- 嘗試載入 GNN API（可選）----
try:
    # 若你的 gnn_api 不在 python path，這裡自行調整/刪除
    import sys
    sys.path.append('/root/ruan_workspace/ic_cad/gnn')
    from gnn_api import load_encoder, get_embeddings
    _HAS_GNN = True
except Exception as _e:
    logger.warning(f"GNN API not available, will fallback to zeros. Detail: {_e}")
    load_encoder = None
    get_embeddings = None
    _HAS_GNN = False


# -------- State --------
@dataclass
class EnvironmentState:
    # 1. 候選集合 S (candidate cells)
    candidate_cells: List[str]  # cell types for CellReplacementManager  
    candidate_instances: List[str]  # corresponding instance names for OpenROAD
    
    # 2. PPO 代理期望的特徵格式
    candidate_gnn_features: List[np.ndarray]     # GNN features for candidates
    candidate_dynamic_features: List[np.ndarray] # Dynamic features for candidates
    global_features: np.ndarray                  # Global state features
    action_mask: Dict[str, np.ndarray]           # Action validity masks
    
    # 3. 靜態特徵 (GNN embeddings, computed once per reset)
    node_emb_full: Optional[np.ndarray]  # [N, D] - 全圖 GNN embeddings
    
    # 4. 動態特徵 (updated every step)
    node_dyn: Optional[np.ndarray]  # [N, F_dyn] - 每個 cell 的動態特徵
    
    # 5. 訓練所需的狀態（必須有預設值）
    current_tns: float = 0.0
    current_wns: float = 0.0  
    current_power: float = 0.0
    initial_tns: float = 0.0
    initial_wns: float = 0.0  
    initial_power: float = 0.0
    step_count: int = 0
    max_steps: int = 20
    done: bool = False
    
    # Helper fields
    total_cells: int = 0  # 總 cell 數量 N
    cell_name_to_idx: Optional[Dict[str, int]] = None  # instance name -> index mapping
    
    def __post_init__(self):
        if self.cell_name_to_idx is None:
            self.cell_name_to_idx = {}


# -------- Environment --------
class OptimizationEnvironment:
    """
    強化學習的環境封裝，用 OpenROAD 做動作與量測
    """
    def __init__(self, config: RLConfig, openroad: Optional[OpenRoadInterface] = None):
        self.config = config
        
        # Initialize Cell Replacement Manager 
        cell_groups_json = getattr(config, "cell_groups_path", "cell_groups.json")
        self.cell_replacement_manager = CellReplacementManager(cell_groups_json)
        
        self.openroad = openroad or OpenRoadInterface(
            work_dir=getattr(config, "openroad_work_dir", "/tmp/openroad_work"),
            pdk_root=getattr(config, "openroad_pdk_root", "~/solution/testcases/ASAP7"),
            design_root=getattr(config, "openroad_design_root", "~/solution/testcases"),
            max_buffer_percent=getattr(config, "max_buffer_percent", 10.0),
            auto_repair_each_step=getattr(config, "auto_repair_each_step", True),
            cell_groups_json=cell_groups_json
        )

        # GNN 相關 - 使用自動路徑檢測
        self._gnn_encoder = None
        self._gnn_meta = None
        # 使用 GNN 目錄的完整路徑
        gnn_dir = "/root/ruan_workspace/ic_cad/gnn"
        self.gnn_meta_path  = getattr(self.config, "gnn_meta_path",  f"{gnn_dir}/encoder_meta.json")
        # 不再硬編碼 gnn_model_path，讓 load_encoder 自動從 meta 中讀取
        self.gnn_model_path = getattr(self.config, "gnn_model_path", None)  # None 表示自動檢測
        
        # 動態讀取 GNN 維度配置，而不是從 config 中硬編碼
        self.gnn_embed_dim = self._get_gnn_embed_dim_from_meta()

        # 互動相關
        self.current_case: Optional[str] = None
        self.current_state: Optional[EnvironmentState] = None

        # 全域初始狀態追蹤 (第一次重置時記錄的最初電路狀態)
        self.global_initial_tns: Optional[float] = None
        self.global_initial_wns: Optional[float] = None
        self.global_initial_power: Optional[float] = None

        # 動作空間設定
        self.action_library: List[Dict[str, Any]] = getattr(config, "action_library", [])
        self.allowed_action_types: Tuple[str, ...] = ("replace_cell",)  # 只使用 replace_cell

        self.action_space_size: int = getattr(config, "action_space_size", 0)
        if self.action_space_size <= 0:
            self.action_space_size = len(self.action_library) if self.action_library else 2
        elif self.action_library:
            self.action_space_size = len(self.action_library)

        self.max_steps_per_episode: int = getattr(config, "max_steps_per_episode", 20)

        # reward 權重（正向加分：越好越大）
        self.reward_weight_tns: float = getattr(config, "reward_weight_tns", 1.0)
        self.reward_weight_wns: float = getattr(config, "reward_weight_wns", 0.5)
        self.reward_weight_power: float = getattr(config, "reward_weight_power", 0.1)
        self.fail_penalty: float = getattr(config, "fail_penalty", 0.3)
        self.tns_goal_eps: float = getattr(config, "tns_goal_eps", 0.05)  # 更積極的閾值
        self.goal_bonus: float = getattr(config, "goal_bonus", 2.0)  # 增加達成目標獎勵
        
        # 混合獎勵機制參數
        self.use_hybrid_reward: bool = getattr(config, "use_hybrid_reward", True)
        self.relative_weight: float = getattr(config, "relative_weight", 0.6)  # 相對改善權重
        self.absolute_weight: float = getattr(config, "absolute_weight", 0.4)  # 絕對改善權重
        self.exploration_weight: float = getattr(config, "exploration_weight", 0.0)  # 探索獎勵權重
        
        # 自適應獎勵調整
        self.adaptive_weights: bool = getattr(config, "adaptive_weights", False)
        self.training_progress: float = 0.0  # 訓練進度 [0, 1]
        self.recent_improvements: List[float] = []  # 記錄最近的改善情況
        self.improvement_history_size: int = getattr(config, "improvement_history_size", 50)
        # 候選集合設定 - 更聚焦timing關鍵路徑
        self.top_slack_candidates: int = getattr(config, "top_slack_candidates", 60)
        self.top_power_candidates: int = getattr(config, "top_power_candidates", 20)

    # ---- Public API ----
    def reset(self, case_name: str) -> EnvironmentState:
        """
        一次載入設計，讀初始報告；之後 step() 都操作同一個 session。
        """
        self.current_case = case_name
        self.openroad.load_case(case_name)

        # 初始報告
        initial_report: MetricsReport = self.openroad.report_metrics(case_name)
        
        # 記錄全域初始狀態 (第一次重置時的電路狀態)
        if self.global_initial_tns is None:
            self.global_initial_tns = initial_report.tns
            self.global_initial_wns = initial_report.wns
            self.global_initial_power = initial_report.total_power
            logger.info(f"🎯 環境記錄最初電路狀態: TNS={self.global_initial_tns:.1f}ns, "
                       f"WNS={self.global_initial_wns:.1f}ns, Power={self.global_initial_power:.6f}W")
        
        # 更新 cell information
        self.openroad.update_cell_information(case_name)
        
        # 獲取靜態特徵 (GNN embeddings)
        node_emb_full = self._get_gnn_embeddings(case_name)
        
        # 獲取動態特徵和 cell 映射
        node_dyn, cell_names = self.openroad.get_dynamic_features(case_name)
        cell_name_to_idx = {name: idx for idx, name in enumerate(cell_names)}
        
        # 獲取候選集合 (instance_name, cell_type) 對
        candidate_pairs = self.openroad.get_candidate_cells(
            case_name, self.top_slack_candidates, self.top_power_candidates
        )
        
        # 分離 instance names 和 cell types
        candidate_instances = [pair[0] for pair in candidate_pairs]
        candidate_cells = [pair[1] for pair in candidate_pairs]
        
        # 生成二維動作空間資訊
        candidate_group_indices = []
        action_mask = []
        
        for cell_type in candidate_cells:
            replacement_options = self.cell_replacement_manager.get_replacement_options(cell_type)
            if replacement_options:
                # 找到對應的 group 索引
                group_idx = None
                for i, group in enumerate(self.cell_replacement_manager.cell_groups):
                    if cell_type in group:
                        group_idx = i
                        break
                
                if group_idx is not None:
                    candidate_group_indices.append(group_idx)
                    # 創建此候選 cell 的動作 mask
                    group_mask = [True] * len(replacement_options)
                    action_mask.append(group_mask)
                else:
                    # 找不到對應群組，使用預設值
                    candidate_group_indices.append(-1)
                    action_mask.append([False])  # 沒有有效替換選項
            else:
                candidate_group_indices.append(-1)
                action_mask.append([False])
        
        # 填充 action_mask 到統一長度
        max_replacements = max(len(mask) for mask in action_mask) if action_mask else 1
        print(f"DEBUG: Environment max_replacements: {max_replacements}")
        for mask in action_mask:
            while len(mask) < max_replacements:
                mask.append(False)

        # 準備 PPO 代理期望的特徵格式
        candidate_gnn_features = []
        candidate_dynamic_features = []
        
        for instance_name in candidate_instances:
            if instance_name in cell_name_to_idx:
                cell_idx = cell_name_to_idx[instance_name]
                # GNN 特徵
                if node_emb_full is not None and cell_idx < len(node_emb_full):
                    gnn_feat = node_emb_full[cell_idx]
                else:
                    gnn_feat = np.random.randn(self.gnn_embed_dim)  # 使用配置的維度
                candidate_gnn_features.append(gnn_feat)
                
                # 動態特徵
                if node_dyn is not None and cell_idx < len(node_dyn):
                    dyn_feat = node_dyn[cell_idx]
                else:
                    dyn_feat = np.random.randn(12)  # 預設動態特徵維度
                candidate_dynamic_features.append(dyn_feat)
            else:
                # 找不到 cell，使用預設特徵
                candidate_gnn_features.append(np.random.randn(self.gnn_embed_dim))  # 使用配置的維度
                candidate_dynamic_features.append(np.random.randn(12))
        
        # 全域特徵
        global_features = np.array([
            initial_report.tns, initial_report.wns, initial_report.total_power,
            0, 0, 0, 0, 0, 0  # 額外的全域特徵，總共 9 維以匹配 config.global_feature_dim
        ])
        
        # 動作遮罩格式轉換 - 修正二維動作空間遮罩
        max_candidates = getattr(self.config, 'max_candidates', 20)
        candidate_mask = np.zeros(max_candidates)
        candidate_mask[:len(candidate_cells)] = 1.0  # 實際候選設為可用
        
        # 為二維動作空間構建正確的替換遮罩
        # 對於每個候選項，檢查是否有有效的替換選項
        replacement_mask = np.zeros(max_replacements)
        if len(candidate_cells) > 0:
            # 取所有候選項中最大的替換選項數量作為參考
            for i, cell_name in enumerate(candidate_cells):
                replacement_options = self.cell_replacement_manager.get_replacement_options(cell_name)
                if replacement_options and len(replacement_options) > 0:
                    # 至少有一個候選項有替換選項，所以第一個替換索引是有效的
                    replacement_mask[0] = 1.0
                    break
        
        action_mask_dict = {
            'candidate_mask': candidate_mask,
            'replacement_mask': replacement_mask,
            'candidate_replacement_masks': action_mask  # 保存每個候選項的詳細遮罩
        }

        self.current_state = EnvironmentState(
            candidate_cells=candidate_cells,
            candidate_instances=candidate_instances,
            candidate_gnn_features=candidate_gnn_features,
            candidate_dynamic_features=candidate_dynamic_features,
            global_features=global_features,
            action_mask=action_mask_dict,  # 使用正確的字段名稱
            node_emb_full=node_emb_full,
            node_dyn=node_dyn,
            current_tns=initial_report.tns,
            current_wns=initial_report.wns,
            current_power=initial_report.total_power,
            initial_tns=initial_report.tns,
            initial_wns=initial_report.wns,
            initial_power=initial_report.total_power,
            step_count=0,
            max_steps=self.max_steps_per_episode,
            done=False,
            total_cells=len(cell_names),
            cell_name_to_idx=cell_name_to_idx
        )
        return self.current_state

    def step(self, action: Tuple[int, int]) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        """
        二維動作 -> 轉為 OptimizationAction -> OpenROAD.apply_action -> report_metrics -> reward
        
        Args:
            action: (candidate_idx, replacement_idx) - 候選 cell 索引和替換選項索引
        """
        if self.current_state is None or self.current_case is None:
            raise RuntimeError("環境尚未重置（請先呼叫 reset(case_name)）")

        candidate_idx, replacement_idx = action
        
        # 1) 驗證動作有效性
        if candidate_idx >= len(self.current_state.candidate_cells):
            raise ValueError(f"Candidate index {candidate_idx} 超出候選範圍 {len(self.current_state.candidate_cells)}")
            
        # 獲取具體的替換選項來驗證 replacement_idx
        target_cell_type = self.current_state.candidate_cells[candidate_idx]
        target_instance = self.current_state.candidate_instances[candidate_idx]
        replacement_options = self.cell_replacement_manager.get_replacement_options(target_cell_type)
        
        # 安全修正：如果替換索引超出範圍，自動修正到有效範圍
        if replacement_idx >= len(replacement_options):
            logger.warning(f"⚠️  替換索引 {replacement_idx} 超出範圍 {len(replacement_options)}，使用 mod 運算修正到 {replacement_idx % len(replacement_options)}")
            replacement_idx = replacement_idx % len(replacement_options)

        target_replacement = replacement_options[replacement_idx]
        optimization_action = OptimizationAction(
            action_type="replace_cell",
            target_cell=target_instance,  # 使用 instance name
            new_cell_type=target_replacement
        )

        # 2) 對 OpenROAD 執行動作（不重載設計）
        success = self.openroad.apply_action(self.current_case, optimization_action)

        # 3) 量測最新指標
        new_report: MetricsReport = self.openroad.report_metrics(self.current_case)


        # 4) 更新動態特徵和候選集合
        self._update_dynamic_state()

        # 5) 計算 reward（正向加分：改善越多分數越高）
        reward = self._calculate_reward(
            old_tns=self.current_state.current_tns,
            new_tns=new_report.tns,
            old_wns=self.current_state.current_wns,
            new_wns=new_report.wns,
            old_power=self.current_state.current_power,
            new_power=new_report.total_power,
            success=success,
        )

        # 6) 更新狀態 - 包括 PPO 代理期望的特徵
        self.current_state.current_tns = new_report.tns
        self.current_state.current_wns = new_report.wns
        self.current_state.current_power = new_report.total_power
        self.current_state.step_count += 1
        
        # 更新全域特徵
        self.current_state.global_features = np.array([
            new_report.tns, new_report.wns, new_report.total_power,
            self.current_state.step_count, 0, 0, 0, 0, 0  # 額外的全域特徵，總共 9 維
        ])

        # 7) 終止條件
        done = False
        if self.current_state.step_count >= self.current_state.max_steps:
            done = True
        if new_report.tns >= -self.tns_goal_eps:
            done = True
        self.current_state.done = done

        info = {
            "success": success,
            "tns": new_report.tns,
            "wns": new_report.wns,
            "power": new_report.total_power,
            "action": optimization_action,
        }
        return self.current_state, float(reward), bool(done), info

    def get_candidate_action_space_size(self) -> int:
        """獲取當前候選動作空間大小"""
        if self.current_state is None:
            return 0
        return len(self.current_state.candidate_cells)
    
    def get_action_space_shape(self) -> Tuple[int, int]:
        """獲取二維動作空間的形狀 (n_candidates, max_replacements)"""
        if self.current_state is None:
            return (0, 0)
        return (len(self.current_state.candidate_cells), self.current_state.max_replacements)
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """獲取所有有效的動作對 (candidate_idx, replacement_idx)"""
        if self.current_state is None:
            return []
        
        valid_actions = []
        for candidate_idx in range(len(self.current_state.candidate_cells)):
            target_cell = self.current_state.candidate_cells[candidate_idx]
            replacement_options = self.cell_replacement_manager.get_replacement_options(target_cell)
            for replacement_idx in range(len(replacement_options)):
                valid_actions.append((candidate_idx, replacement_idx))
        return valid_actions

    def get_state_vector(self) -> np.ndarray:
        """
        回傳狀態向量，現在包含：
        - 候選 cell 的 GNN 特徵 (平均 pooling)
        - 候選 cell 的動態特徵 (平均 pooling) 
        - 全域特徵
        """
        if self.current_state is None:
            raise RuntimeError("環境尚未重置")

        # 1) 候選 cells 的 GNN 特徵
        candidate_gnn_features = np.zeros(self.gnn_embed_dim, dtype=np.float32)
        if (self.current_state.node_emb_full is not None and 
            self.current_state.candidate_cells):
            # 使用 candidate_cells 名單來獲取對應的 embeddings
            candidate_indices = []
            for cell_name in self.current_state.candidate_cells:
                if cell_name in self.current_state.cell_name_to_idx:
                    candidate_indices.append(self.current_state.cell_name_to_idx[cell_name])
            
            if candidate_indices:
                candidate_embeddings = self.current_state.node_emb_full[candidate_indices]
                candidate_gnn_features = np.mean(candidate_embeddings, axis=0).astype(np.float32)
                # 對齊維度
                if candidate_gnn_features.shape[0] < self.gnn_embed_dim:
                    pad = np.zeros(self.gnn_embed_dim - candidate_gnn_features.shape[0], dtype=np.float32)
                    candidate_gnn_features = np.concatenate([candidate_gnn_features, pad])
                elif candidate_gnn_features.shape[0] > self.gnn_embed_dim:
                    candidate_gnn_features = candidate_gnn_features[:self.gnn_embed_dim]
        
        # 2) 候選 cells 的動態特徵
        candidate_dyn_features = np.zeros(5, dtype=np.float32)  # 假設取 5 個主要動態特徵
        if (self.current_state.node_dyn is not None and 
            self.current_state.candidate_cells):
            # 使用 candidate_cells 名單來獲取對應的動態特徵
            candidate_indices = []
            for cell_name in self.current_state.candidate_cells:
                if cell_name in self.current_state.cell_name_to_idx:
                    candidate_indices.append(self.current_state.cell_name_to_idx[cell_name])
            
            if candidate_indices:
                candidate_dynamics = self.current_state.node_dyn[candidate_indices]
                # 取一些關鍵特徵的統計值
                candidate_dyn_features = np.array([
                    np.mean(candidate_dynamics[:, 0]),  # avg worst slack
                    np.mean(candidate_dynamics[:, 1]),  # avg power
                    np.std(candidate_dynamics[:, 0]),   # slack diversity  
                    np.max(candidate_dynamics[:, 1]),   # max power
                    len(candidate_dynamics)             # candidate count
                ], dtype=np.float32)

        # 3) 全域特徵
        global_features = np.array([
            self.current_state.current_tns / max(1.0, abs(self.current_state.initial_tns)),
            self.current_state.current_wns / max(1.0, abs(self.current_state.initial_wns)),
            self.current_state.current_power / max(1e-12, self.current_state.initial_power),
            self.current_state.step_count / max(1, self.current_state.max_steps),
            len(self.current_state.candidate_cells) / max(1, self.current_state.total_cells),
            float(self.current_state.done),
            1.0 if self.current_state.current_tns >= -self.tns_goal_eps else 0.0,
            0.0, 0.0  # 額外兩個特徵以達到 9 維
        ], dtype=np.float32)

        # 合併所有特徵
        state_vector = np.concatenate([candidate_gnn_features, candidate_dyn_features, global_features])
        return state_vector

    # ---- Internals ----
    def _update_dynamic_state(self):
        """更新動態特徵和候選集合（重新選擇最關鍵的候選）"""
        if self.current_case is None:
            return
            
        # 更新 cell information
        self.openroad.update_cell_information(self.current_case)
        
        # 更新動態特徵
        node_dyn, cell_names = self.openroad.get_dynamic_features(self.current_case)
        cell_name_to_idx = {name: idx for idx, name in enumerate(cell_names)}
        self.current_state.node_dyn = node_dyn
        self.current_state.cell_name_to_idx = cell_name_to_idx
        
        # 🎯 重新選擇候選集合 - 根據最新的時序分析結果
        candidate_pairs = self.openroad.get_candidate_cells(
            self.current_case, self.top_slack_candidates, self.top_power_candidates
        )
        
        # 分離 instance names 和 cell types
        candidate_instances = [pair[0] for pair in candidate_pairs]
        candidate_cells = [pair[1] for pair in candidate_pairs]
        
        # 更新候選集合
        self.current_state.candidate_instances = candidate_instances
        self.current_state.candidate_cells = candidate_cells
        
        # 重新計算候選集合的動作 mask
        new_action_mask = []
        for cell_name in candidate_cells:
            replacement_options = self.cell_replacement_manager.get_replacement_options(cell_name)
            if replacement_options:
                # 可以在這裡加入額外的有效性檢查，例如：
                # - 是否已經是最優的 cell type
                # - 是否有足夠的 slack margin 進行替換
                group_mask = [True] * len(replacement_options)
                new_action_mask.append(group_mask)
            else:
                new_action_mask.append([False])
        
        # 填充到統一長度
        max_replacements = max(len(mask) for mask in new_action_mask) if new_action_mask else 1
        for mask in new_action_mask:
            while len(mask) < max_replacements:
                mask.append(False)
        
        self.current_state.action_mask = new_action_mask
        self.current_state.max_replacements = max_replacements
        
        # 🎯 重新計算候選 cell 的 GNN 和動態特徵
        self._update_candidate_features()
        
        # logger.info(f"📋 候選集合已更新: {len(candidate_cells)} 個候選 cell (TNS: {self.current_state.current_tns:.1f}ns)")

    def _update_candidate_features(self):
        """重新計算候選 cell 的 GNN 和動態特徵"""
        if not self.current_state.candidate_instances:
            return
            
        # 重新計算候選 cell 的 GNN 特徵
        candidate_gnn_features = []
        candidate_dynamic_features = []
        
        for instance_name in self.current_state.candidate_instances:
            if instance_name in self.current_state.cell_name_to_idx:
                cell_idx = self.current_state.cell_name_to_idx[instance_name]
                
                # GNN 特徵
                if self.current_state.node_emb_full is not None:
                    gnn_features = self.current_state.node_emb_full[cell_idx]
                    candidate_gnn_features.append(gnn_features)
                
                # 動態特徵
                if self.current_state.node_dyn is not None:
                    dyn_features = self.current_state.node_dyn[cell_idx]
                    candidate_dynamic_features.append(dyn_features)
        
        # 更新候選特徵到狀態
        if candidate_gnn_features:
            self.current_state.candidate_gnn_features = np.array(candidate_gnn_features)
        if candidate_dynamic_features:
            self.current_state.candidate_dynamic_features = np.array(candidate_dynamic_features)

    def _decode_action_for_cell(self, target_cell: str) -> OptimizationAction:
        """
        為特定 cell 創建優化動作
        這裡可以根據 cell 的特性選擇適當的動作類型
        """
        # 簡單版本：預設使用 replace_cell
        return OptimizationAction(
            action_type="replace_cell",
            target_cell=target_cell
        )

    def _decode_action(self, action_idx: int) -> OptimizationAction:
        """
        舊版動作解碼方法 - 現在主要使用 _decode_action_for_cell
        保留此方法作為向後相容性
        """
        # 1) 使用 action_library（建議）
        if self.action_library:
            spec = self.action_library[action_idx % len(self.action_library)]
            a_type = spec.get("action_type", "replace_cell")
            if a_type not in self.allowed_action_types:
                a_type = "replace_cell"

            return OptimizationAction(
                action_type=a_type,
                target_cell=spec.get("target_cell", ""),
                new_cell_type=spec.get("new_cell_type"),
                position=spec.get("position"),
            )

        # 2) 範例動作 - 統一使用 replace_cell
        target_cell = getattr(self.config, "demo_target_cell", "_1_")
        new_master = getattr(self.config, "demo_new_master", "NAND2xp33_ASAP7_75t_L")
        return OptimizationAction(action_type="replace_cell",
                                  target_cell=target_cell,
                                  new_cell_type=new_master)

    def _calculate_reward(
        self,
        old_tns: float,
        new_tns: float,
        old_wns: float,
        new_wns: float,
        old_power: float,
        new_power: float,
        success: bool,
    ) -> float:
        """
        革新的獎勵機制：使用獎勵塑形(Reward Shaping)和基線獎勵
        重點：讓代理從小的改善中學習，避免持續負獎勵
        """
        if not success:
            return -1.0  # 減輕失敗懲罰

        # 獲取全域初始狀態作為基準
        if self.global_initial_tns is not None:
            global_initial_tns = self.global_initial_tns
            global_initial_wns = self.global_initial_wns
            global_initial_power = self.global_initial_power
        else:
            global_initial_tns = getattr(self.current_state, "initial_tns", old_tns)
            global_initial_wns = getattr(self.current_state, "initial_wns", old_wns)
            global_initial_power = getattr(self.current_state, "initial_power", old_power)

        # 🎯 核心修正：TNS和WNS是負值，越接近0越好
        # 改善意味著絕對值減小
        tns_improvement = abs(old_tns) - abs(new_tns)  # 正值表示TNS改善(絕對值減小)
        wns_improvement = abs(old_wns) - abs(new_wns)  # 正值表示WNS改善(絕對值減小)
        power_improvement = old_power - new_power      # 正值表示功耗減少

        # 🔄 移除誤導性的基線獎勵，改用真實的改善獎勵
        base_reward = 0.0  # 不給無條件獎勵

        # 📈 對稱的獎勵/懲罰機制 - 修正版（處理除零）
        step_reward = 0.0
        # TNS獎勵：改善和惡化使用相同權重，處理除零情況
        if abs(global_initial_tns) > 1e-6:  # 有意義的初始TNS值
            if tns_improvement > 0:  # TNS真正改善了
                step_reward += 5.0 * (tns_improvement / abs(global_initial_tns))
            else:  # TNS惡化了
                step_reward += 5.0 * (tns_improvement / abs(global_initial_tns))  # 等權重負獎勵
        else:  # 初始TNS接近0（組合邏輯電路），使用絕對改善值
            if abs(tns_improvement) > 1e-6:  # 有意義的TNS變化
                step_reward += 1.0 if tns_improvement > 0 else -1.0
            
        # WNS獎勵：改善和惡化使用相同權重，處理除零情況
        if abs(global_initial_wns) > 1e-6:  # 有意義的初始WNS值
            if wns_improvement > 0:  # WNS真正改善了
                step_reward += 3.0 * (wns_improvement / abs(global_initial_wns))
            else:  # WNS惡化了
                step_reward += 3.0 * (wns_improvement / abs(global_initial_wns))  # 等權重負獎勵
        else:  # 初始WNS接近0，使用絕對改善值
            if abs(wns_improvement) > 1e-6:  # 有意義的WNS變化
                step_reward += 0.5 if wns_improvement > 0 else -0.5
            
        # Power獎勵：改善和惡化使用相同權重，處理除零情況
        if abs(global_initial_power) > 1e-9:  # 有意義的初始功耗值
            if power_improvement > 0:  # 功耗減少了
                step_reward += 1.0 * (power_improvement / global_initial_power)
            else:  # 功耗增加了
                step_reward += 1.75 * (power_improvement / global_initial_power)  # 等權重負獎勵
        else:  # 初始功耗接近0（不太可能但防禦性編程）
            if abs(power_improvement) > 1e-9:
                step_reward += 0.1 if power_improvement > 0 else -0.1

        # 🎖️ 里程碑獎勵：只獎勵相對於初始電路的改善，處理除零情況
        milestone_reward = 0.0
        global_tns_improvement = abs(global_initial_tns) - abs(new_tns)  # 相對初始的改善
        global_wns_improvement = abs(global_initial_wns) - abs(new_wns)  # 相對初始的改善
        global_power_improvement = abs(global_initial_power) - abs(new_power)

        if global_tns_improvement > 0 and abs(global_initial_tns) > 1e-6:  # 相對初始狀態有改善且有意義
            improvement_ratio = global_tns_improvement / abs(global_initial_tns)
            if improvement_ratio > 0.05:  # 5%以上改善
                milestone_reward += 10.0
            elif improvement_ratio > 0.01:  # 1%以上改善
                milestone_reward += 5.0
        elif global_tns_improvement > 0 and abs(global_initial_tns) <= 1e-6:  # 組合邏輯電路的改善
            # 對於無時序約束的電路，任何有意義的TNS改善都給小獎勵
            if abs(global_tns_improvement) > 1e-6:
                milestone_reward += 2.0
        elif global_tns_improvement < 0 and abs(global_initial_tns) > 1e-6:  # 相對初始狀態惡化且有意義
            degradation_ratio = abs(global_tns_improvement) / abs(global_initial_tns)
            if degradation_ratio > 0.2:  # 惡化超過20%
                milestone_reward -= 5.0
        elif global_tns_improvement < 0 and abs(global_initial_tns) <= 1e-6:  # 組合邏輯電路的惡化
            # 對於無時序約束的電路，任何有意義的TNS惡化都給小懲罰
            if abs(global_tns_improvement) > 1e-6:
                milestone_reward -= 1.0

        if global_power_improvement > 0 and abs(global_initial_power) > 1e-9:  # 相對初始狀態有改善且有意義
            improvement_ratio = global_power_improvement / abs(global_initial_power)
            if improvement_ratio > 0.05:  # 5%以上改善
                milestone_reward += 10.0
            elif improvement_ratio > 0.01:  # 1%以上改善
                milestone_reward += 5.0
        elif global_power_improvement < 0 and abs(global_initial_power) > 1e-9:  # 相對初始狀態惡化且有意義
            degradation_ratio = abs(global_power_improvement) / abs(global_initial_power)
            if degradation_ratio > 0.2:  # 惡化超過20%
                milestone_reward -= 10.0

        # 🚫 移除額外懲罰機制，讓自然負獎勵發揮作用
        penalty = 0.0
        # 原本的step_reward已經包含適當的負獎勵，不需要額外懲罰

        # 🏆 最終獎勵組合
        total_reward = base_reward + step_reward + milestone_reward - penalty

        # 🔧 對稱的獎勵範圍：允許充分的負獎勵來引導學習
        reward = np.clip(total_reward, -15.0, 15.0)

        return float(reward)

    def _get_current_weights(self) -> Tuple[float, float, float]:
        """
        獲取當前的獎勵權重，支援自適應調整
        Returns: (relative_weight, absolute_weight, exploration_weight)
        """
        if not self.adaptive_weights:
            return self.relative_weight, self.absolute_weight, self.exploration_weight

        # 根據訓練進度自適應調整
        progress = self.training_progress
        
        if progress < 0.3:  # 早期探索階段
            rel_weight = 0.7
            abs_weight = 0.3
            exp_weight = 0.1
        elif progress < 0.7:  # 穩定優化階段
            rel_weight = 0.5
            abs_weight = 0.5
            exp_weight = 0.05
        else:  # 精細調整階段
            rel_weight = 0.3
            abs_weight = 0.6
            exp_weight = 0.1

        # 檢測停滯並調整
        if self._detect_stagnation():
            rel_weight += 0.2  # 增加對相對改善的重視
            exp_weight *= 2.0  # 增加探索獎勵

        return rel_weight, abs_weight, exp_weight

    def _calculate_exploration_bonus(self) -> float:
        """
        計算探索獎勵，鼓勵動作多樣性
        這裡可以根據實際需求實現，例如：
        - 基於動作選擇的熵
        - 基於訪問過的狀態數量
        - 基於動作序列的多樣性
        """
        # 簡單實現：隨機探索獎勵
        # 在實際應用中，這裡可以實現更複雜的探索度量
        return 0.01  # 小的固定探索獎勵

    def _update_improvement_history(self, improvement: float):
        """更新改善歷史記錄"""
        self.recent_improvements.append(improvement)
        if len(self.recent_improvements) > self.improvement_history_size:
            self.recent_improvements.pop(0)

    def _detect_stagnation(self) -> bool:
        """
        檢測是否進入停滯狀態
        如果最近的改善都很小，則認為進入停滯
        """
        if len(self.recent_improvements) < 20:  # 需要足夠的歷史數據
            return False
        
        recent_window = self.recent_improvements[-20:]
        avg_improvement = sum(recent_window) / len(recent_window)
        
        # 如果平均改善很小，認為停滯
        return abs(avg_improvement) < 0.001

    def update_training_progress(self, progress: float):
        """
        更新訓練進度，用於自適應權重調整
        Args:
            progress: 訓練進度 [0.0, 1.0]
        """
        self.training_progress = max(0.0, min(1.0, progress))

    # ---- 真正載入/計算 GNN Embeddings ----
    def _get_gnn_embeddings(self, case_name: str) -> Optional[np.ndarray]:
        """
        回傳 [num_nodes, hidden] 的 numpy array；若 GNN 不可用則回 None。
        """
        if not _HAS_GNN:
            return None
        try:
            if self._gnn_encoder is None:
                self._gnn_encoder, self._gnn_meta = load_encoder(
                    model_path=self.gnn_model_path,
                    meta_path=self.gnn_meta_path
                )
            emb = get_embeddings(case_name, self._gnn_encoder)  # torch.Tensor [N, D]
            return emb.detach().cpu().numpy()
        except Exception as e:
            logger.warning(f"GNN embeddings failed for {case_name}, fallback to zeros. Detail: {e}")
            return None

    def _get_gnn_embed_dim_from_meta(self) -> int:
        """
        從 encoder_meta.json 動態讀取 GNN 嵌入維度
        """
        import json
        import os
        
        try:
            if os.path.exists(self.gnn_meta_path):
                with open(self.gnn_meta_path, 'r') as f:
                    meta = json.load(f)
                embed_dim = meta.get('hidden_dim', 128)  # 預設 128
                logger.info(f"📏 從 meta 檔案讀取 GNN 嵌入維度: {embed_dim}")
                return embed_dim
            else:
                logger.warning(f"⚠️ Meta 檔案不存在: {self.gnn_meta_path}, 使用預設維度 128")
                return 128
        except Exception as e:
            logger.warning(f"⚠️ 讀取 meta 檔案失敗: {e}, 使用預設維度 128")
            return 128


# -------- Factory --------
def create_environment(config: RLConfig) -> OptimizationEnvironment:
    env = OptimizationEnvironment(config=config)
    logger.info("Environment created: OpenROAD-backed optimization environment is ready.")
    return env
