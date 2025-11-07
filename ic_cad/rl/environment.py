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
    
    # 2. 全電路特徵 (新設計：包含所有cells的完整信息)
    all_cell_gnn_features: Optional[np.ndarray]     # [N_total, D] - 全電路GNN特徵
    all_cell_dynamic_features: Optional[np.ndarray] # [N_total, F_dyn] - 全電路動態特徵
    all_cell_names: List[str]                       # 所有cell的名稱列表，與特徵對應
    candidate_indices: List[int]                    # 候選cells在全電路中的索引
    
    # 3. PPO 代理期望的特徵格式 (保持向後兼容)
    candidate_gnn_features: List[np.ndarray]     # GNN features for candidates (從全電路提取)
    candidate_dynamic_features: List[np.ndarray] # Dynamic features for candidates (從全電路提取)
    global_features: np.ndarray                  # Global state features
    action_mask: Dict[str, np.ndarray]           # Action validity masks
    
    # 4. 靜態特徵 (GNN embeddings, computed once per reset)
    node_emb_full: Optional[np.ndarray]  # [N, D] - 全圖 GNN embeddings (別名，指向all_cell_gnn_features)
    
    # 5. 動態特徵 (updated every step)
    node_dyn: Optional[np.ndarray]  # [N, F_dyn] - 每個 cell 的動態特徵 (別名，指向all_cell_dynamic_features)
    
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
    max_replacements: int = 84  # 最大替換選項數量（與配置保持一致）
    
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
        
        # 🎯 自動檢測並更新 max_replacements 配置
        recommended_max_replacements = self.cell_replacement_manager.get_recommended_max_replacements()
        config_max_replacements = getattr(config, 'max_replacements', 84)
        
        if recommended_max_replacements != config_max_replacements:
            logger.info(f"🔧 自動更新 max_replacements: {config_max_replacements} -> {recommended_max_replacements}")
            # 動態更新配置
            if hasattr(config, 'max_replacements'):
                config.max_replacements = recommended_max_replacements
            self.auto_max_replacements = recommended_max_replacements
        else:
            self.auto_max_replacements = config_max_replacements
            logger.info(f"✅ max_replacements 配置正確: {config_max_replacements}")
        
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

        self.max_steps_per_episode: int = getattr(config, "max_steps_per_episode", 20)

        # reward 權重（正向加分：越好越大）
        self.reward_weight_tns: float = getattr(config, "reward_weight_tns", 1.0)
        self.reward_weight_wns: float = getattr(config, "reward_weight_wns", 0.5)
        self.reward_weight_power: float = getattr(config, "reward_weight_power", 0.1)
        self.fail_penalty: float = getattr(config, "fail_penalty", 0.3)
        self.tns_goal_eps: float = getattr(config, "tns_goal_eps", 0.05)  # 更積極的閾值
        self.goal_bonus: float = getattr(config, "goal_bonus", 2.0)  # 增加達成目標獎勵
        
        # 動態權重支援
        self.use_dynamic_weights: bool = getattr(config, "use_dynamic_weights", False)
        self.dynamic_tns_weight: float = getattr(config, "dynamic_tns_weight", 1.0)
        self.dynamic_power_weight: float = getattr(config, "dynamic_power_weight", 1.0)
        self.normalize_weights: bool = getattr(config, "normalize_weights", True)
        
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

    def set_dynamic_weights(self, tns_weight: float, power_weight: float):
        """
        設置動態權重
        
        Args:
            tns_weight: TNS 獎勵權重
            power_weight: Power 獎勵權重
        """
        self.dynamic_tns_weight = tns_weight
        self.dynamic_power_weight = power_weight
        self.use_dynamic_weights = True
        
        if self.normalize_weights:
            # 正規化權重，確保總和為1
            total_weight = tns_weight + power_weight
            if total_weight > 0:
                self.dynamic_tns_weight = tns_weight / total_weight
                self.dynamic_power_weight = power_weight / total_weight
    
    def get_current_weights(self) -> Tuple[float, float]:
        """
        獲取當前的 TNS 和 Power 權重
        
        Returns:
            Tuple[float, float]: (tns_weight, power_weight)
        """
        if self.use_dynamic_weights:
            return self.dynamic_tns_weight, self.dynamic_power_weight
        else:
            # 使用預設權重的相對比例
            total = self.reward_weight_tns + self.reward_weight_power
            if total > 0:
                return (self.reward_weight_tns / total, self.reward_weight_power / total)
            else:
                return (0.8, 0.2)  # 預設比例：80% TNS, 20% Power

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
        
        # 填充 action_mask 到統一長度 - 使用自動檢測的值
        max_replacements = self.auto_max_replacements
        dynamic_max = max(len(mask) for mask in action_mask) if action_mask else 1
        
        # 偵測配置不匹配時才記錄警告
        if dynamic_max > max_replacements:
            logger.warning(f"⚠️ 動態計算的 max_replacements ({dynamic_max}) 超出自動檢測限制 ({max_replacements})，將截斷")
        elif dynamic_max > max_replacements * 0.8:  # 接近限制時的調試信息
            logger.debug(f"🔧 動態 max_replacements: {dynamic_max}, 自動檢測限制: {max_replacements}")
        
        # 確保不會超出配置的限制
        
        # 修正每個 mask 到統一長度
        for i, mask in enumerate(action_mask):
            # 先截斷到配置限制
            if len(mask) > max_replacements:
                action_mask[i] = mask[:max_replacements]
            # 然後填充到配置長度
            while len(action_mask[i]) < max_replacements:
                action_mask[i].append(False)

        # 準備全電路特徵 (新設計：包含所有cells)
        all_cell_gnn_features = node_emb_full  # [N_total, 128]
        all_cell_dynamic_features = node_dyn   # [N_total, F_dyn]
        all_cell_names = cell_names            # List[str]
        
        # 計算候選cells在全電路中的索引
        candidate_indices = []
        for instance_name in candidate_instances:
            if instance_name in cell_name_to_idx:
                candidate_indices.append(cell_name_to_idx[instance_name])
            else:
                logger.warning(f"⚠️ 候選cell {instance_name} 未在全電路cell列表中找到")
                candidate_indices.append(-1)  # 標記為無效
        
        # 準備 PPO 代理期望的特徵格式 (保持向後兼容)
        candidate_gnn_features = []
        candidate_dynamic_features = []
        
        for instance_name in candidate_instances:
            if instance_name in cell_name_to_idx:
                cell_idx = cell_name_to_idx[instance_name]
                # GNN 特徵
                if node_emb_full is not None and cell_idx < len(node_emb_full):
                    gnn_feat = node_emb_full[cell_idx]
                else:
                    gnn_feat = np.zeros(self.gnn_embed_dim, dtype=np.float32)  # 使用 zeros 而非 randn，避免噪音
                candidate_gnn_features.append(gnn_feat)
                
                # 動態特徵
                if node_dyn is not None and cell_idx < len(node_dyn):
                    dyn_feat = node_dyn[cell_idx]
                else:
                    dyn_feat = np.zeros(9, dtype=np.float32)  # 使用 zeros 而非 randn，避免噪音
                candidate_dynamic_features.append(dyn_feat)
            else:
                # 找不到 cell，使用零向量而非隨機噪音
                candidate_gnn_features.append(np.zeros(self.gnn_embed_dim, dtype=np.float32))
                candidate_dynamic_features.append(np.zeros(9, dtype=np.float32))
        
        # 全域特徵 (正規化，避免數值尺度問題) - 7 維
        global_features = np.array([
            initial_report.tns / max(1.0, abs(initial_report.tns)) if abs(initial_report.tns) > 1e-6 else 0.0,
            initial_report.wns / max(1.0, abs(initial_report.wns)) if abs(initial_report.wns) > 1e-6 else 0.0,
            initial_report.total_power / max(1e-9, initial_report.total_power) if initial_report.total_power > 1e-9 else 0.0,
            0.0,  # step_count / max_steps (初始為 0)
            float(len(candidate_cells)) / max(1, len(cell_names)),  # 候選比例
            0.0,  # done (初始為 False)
            0.0,  # remaining_steps_ratio (剩餘步數比例)
        ], dtype=np.float32)
        
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
            # 候選集合
            candidate_cells=candidate_cells,
            candidate_instances=candidate_instances,
            
            # 全電路特徵 (新設計)
            all_cell_gnn_features=all_cell_gnn_features,
            all_cell_dynamic_features=all_cell_dynamic_features,
            all_cell_names=all_cell_names,
            candidate_indices=candidate_indices,
            
            # PPO 代理期望的特徵格式 (保持向後兼容)
            candidate_gnn_features=candidate_gnn_features,
            candidate_dynamic_features=candidate_dynamic_features,
            global_features=global_features,
            action_mask=action_mask_dict,
            
            # 靜態和動態特徵 (別名)
            node_emb_full=node_emb_full,
            node_dyn=node_dyn,
            
            # 狀態信息
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
            cell_name_to_idx=cell_name_to_idx,
            max_replacements=max_replacements
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
            logger.error(f"🔥 PPO 代理採樣錯誤:")
            logger.error(f"   候選 cell: {target_cell_type}")
            logger.error(f"   可用替換選項: {replacement_options}")
            logger.error(f"   替換選項數量: {len(replacement_options)}")
            logger.error(f"   PPO 採樣的索引: {replacement_idx}")
            
            # 檢查當前的動作遮罩
            if hasattr(self.current_state, 'action_mask') and self.current_state.action_mask:
                mask_info = self.current_state.action_mask
                if 'candidate_replacement_masks' in mask_info:
                    if candidate_idx < len(mask_info['candidate_replacement_masks']):
                        candidate_mask = mask_info['candidate_replacement_masks'][candidate_idx]
                        valid_count = sum(candidate_mask)
                        logger.error(f"   當前候選 {candidate_idx} 的遮罩: 長度={len(candidate_mask)}, 有效數量={valid_count}")
                        logger.error(f"   遮罩前10個值: {candidate_mask[:10]}")
                    else:
                        logger.error(f"   候選索引 {candidate_idx} 超出遮罩範圍")
                else:
                    logger.error(f"   遮罩中沒有 'candidate_replacement_masks' 鍵")
            else:
                logger.error(f"   當前狀態沒有有效的 action_mask")
            
            logger.error(f"   這表明 PPO 代理的動作遮罩沒有正確工作")
            
            # 使用 mod 運算作為緊急修正
            corrected_idx = replacement_idx % len(replacement_options)
            logger.warning(f"⚠️  使用 mod 運算修正到索引 {corrected_idx} -> {replacement_options[corrected_idx]}")
            replacement_idx = corrected_idx

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
        
        # 更新全域特徵 (正規化，避免數值尺度問題) - 7 維
        remaining_steps_ratio = float(self.current_state.max_steps - self.current_state.step_count) / max(1, self.current_state.max_steps)
        self.current_state.global_features = np.array([
            new_report.tns / max(1.0, abs(self.current_state.initial_tns)) if abs(self.current_state.initial_tns) > 1e-6 else 0.0,
            new_report.wns / max(1.0, abs(self.current_state.initial_wns)) if abs(self.current_state.initial_wns) > 1e-6 else 0.0,
            new_report.total_power / max(1e-9, self.current_state.initial_power) if self.current_state.initial_power > 1e-9 else 0.0,
            float(self.current_state.step_count) / max(1, self.current_state.max_steps),  # 進度比例
            float(len(self.current_state.candidate_cells)) / max(1, self.current_state.total_cells),  # 候選比例
            float(self.current_state.done),  # 是否終止
            remaining_steps_ratio,  # 剩餘步數比例
        ], dtype=np.float32)

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
        candidate_dyn_features = np.zeros(9, dtype=np.float32)  # 現在是 9 維動態特徵
        if (self.current_state.node_dyn is not None and 
            self.current_state.candidate_cells):
            # 使用 candidate_cells 名單來獲取對應的動態特徵
            candidate_indices = []
            for cell_name in self.current_state.candidate_cells:
                if cell_name in self.current_state.cell_name_to_idx:
                    candidate_indices.append(self.current_state.cell_name_to_idx[cell_name])
            
            if candidate_indices:
                candidate_dynamics = self.current_state.node_dyn[candidate_indices]
                # 使用所有 9 維特徵的統計值
                candidate_dyn_features = np.array([
                    np.mean(candidate_dynamics[:, 0]),  # avg total_power
                    np.mean(candidate_dynamics[:, 1]),  # avg delay
                    np.mean(candidate_dynamics[:, 2]),  # avg drive_resistance
                    np.mean(candidate_dynamics[:, 3]),  # avg vt_type
                    np.mean(candidate_dynamics[:, 4]),  # avg fanout_count
                    np.mean(candidate_dynamics[:, 5]),  # avg output_cap
                    np.mean(candidate_dynamics[:, 6]),  # avg output_slew
                    np.mean(candidate_dynamics[:, 7]),  # avg area
                    np.mean(candidate_dynamics[:, 8]),  # avg is_endpoint ratio
                ], dtype=np.float32)

        # 3) 全域特徵 (與 reset/step 中的格式一致，使用 7 維)
        remaining_steps_ratio = float(self.current_state.max_steps - self.current_state.step_count) / max(1, self.current_state.max_steps)
        global_features = np.array([
            self.current_state.current_tns / max(1.0, abs(self.current_state.initial_tns)) if abs(self.current_state.initial_tns) > 1e-6 else 0.0,
            self.current_state.current_wns / max(1.0, abs(self.current_state.initial_wns)) if abs(self.current_state.initial_wns) > 1e-6 else 0.0,
            self.current_state.current_power / max(1e-9, self.current_state.initial_power) if self.current_state.initial_power > 1e-9 else 0.0,
            float(self.current_state.step_count) / max(1, self.current_state.max_steps),
            float(len(self.current_state.candidate_cells)) / max(1, self.current_state.total_cells),
            float(self.current_state.done),
            remaining_steps_ratio,  # 剩餘步數比例
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
        
        # 更新全電路信息
        self.current_state.all_cell_dynamic_features = node_dyn
        self.current_state.all_cell_names = cell_names
        self.current_state.node_dyn = node_dyn  # 別名，保持向後兼容
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
        
        # 重新計算候選cells在全電路中的索引
        candidate_indices = []
        for instance_name in candidate_instances:
            if instance_name in cell_name_to_idx:
                candidate_indices.append(cell_name_to_idx[instance_name])
            else:
                logger.warning(f"⚠️ 候選cell {instance_name} 未在全電路cell列表中找到")
                candidate_indices.append(-1)  # 標記為無效
        self.current_state.candidate_indices = candidate_indices
        
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
        
        # 填充到統一長度 - 使用自動檢測的值
        max_replacements = self.auto_max_replacements
        dynamic_max = max(len(mask) for mask in new_action_mask) if new_action_mask else 1
        
        # 確保不會超出自動檢測的限制
        if dynamic_max > max_replacements:
            logger.warning(f"⚠️ 動態狀態更新: 動態 max_replacements ({dynamic_max}) 超出自動檢測限制 ({max_replacements})")
        
        # 修正每個 mask 到統一長度
        for i, mask in enumerate(new_action_mask):
            # 先截斷到配置限制
            if len(mask) > max_replacements:
                new_action_mask[i] = mask[:max_replacements]
            # 然後填充到配置長度
            while len(new_action_mask[i]) < max_replacements:
                new_action_mask[i].append(False)
        
        self.current_state.action_mask = {
            'candidate_mask': np.ones(max(len(candidate_cells), 1)),  # 所有候選都可用
            'replacement_mask': np.zeros(max_replacements),  # 將在需要時更新
            'candidate_replacement_masks': new_action_mask  # 詳細的每候選遮罩
        }
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
        改良的獎勵機制：分段 + 對數尾部設計
        - 解決獎勵飽和問題：無上限，持續激勵優化
        - 保持不對稱懲罰：改善高獎勵，惡化低懲罰
        - 數值穩定：後期增速放緩，避免梯度爆炸
        """
        if not success:
            return -0.5  # 減輕失敗懲罰

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

        # 獲取當前權重
        tns_weight, power_weight = self.get_current_weights()
        
        # ========================================
        # 📊 步驟獎勵 (Step Reward) - 相對於上一步
        # ========================================
        step_reward = 0.0
        
        # TNS 步驟獎勵：不對稱設計 (8:1)
        if abs(global_initial_tns) > 1e-6:
            tns_ratio = tns_improvement / abs(global_initial_tns)
            if tns_improvement > 0:  # 改善
                step_reward += tns_weight * 8.0 * tns_ratio
            else:  # 惡化 - 輕懲罰
                step_reward += tns_weight * 1.0 * tns_ratio
        else:  # 組合邏輯電路
            if abs(tns_improvement) > 1e-6:
                step_reward += tns_weight * (3.0 if tns_improvement > 0 else -0.2)
            
        # WNS 步驟獎勵：不對稱設計 (4:1)
        if abs(global_initial_wns) > 1e-6:
            wns_ratio = wns_improvement / abs(global_initial_wns)
            if wns_improvement > 0:  # 改善
                step_reward += 4.0 * wns_ratio
            else:  # 惡化 - 輕懲罰
                step_reward += 1.0 * wns_ratio
        else:
            if abs(wns_improvement) > 1e-6:
                step_reward += 1.5 if wns_improvement > 0 else -0.1
            
        # Power 步驟獎勵：不對稱設計 (20:1)
        if abs(global_initial_power) > 1e-9:
            power_ratio = power_improvement / global_initial_power
            if power_improvement > 0:  # 改善
                step_reward += power_weight * 4.0 * power_ratio
            else:  # 惡化 - 輕懲罰
                step_reward += power_weight * 0.2 * power_ratio
        else:
            if abs(power_improvement) > 1e-9:
                step_reward += power_weight * (0.3 if power_improvement > 0 else -0.02)

        # ========================================
        # 🎖️ 里程碑獎勵 (Milestone Reward) - 分段 + 對數尾部
        # ========================================
        milestone_reward = 0.0
        
        # 計算相對於全域初始的改善
        global_tns_improvement = abs(global_initial_tns) - abs(new_tns)
        global_power_improvement = global_initial_power - new_power

        # --- TNS 里程碑獎勵：分段 + 對數尾部設計 ---
        if abs(global_initial_tns) > 1e-6:
            improvement_ratio = global_tns_improvement / abs(global_initial_tns)
            
            if improvement_ratio > 0:  # 有改善
                # 分段獎勵：前期線性，後期對數
                if improvement_ratio < 0.2:  # 0-20%
                    milestone_reward += tns_weight * improvement_ratio * 100  # 最高 20
                elif improvement_ratio < 0.5:  # 20-50%
                    base = tns_weight * 20.0
                    milestone_reward += base + tns_weight * (improvement_ratio - 0.2) * 66.7  # 20 → 40
                elif improvement_ratio < 0.8:  # 50-80%
                    base = tns_weight * 40.0
                    milestone_reward += base + tns_weight * (improvement_ratio - 0.5) * 66.7  # 40 → 60
                else:  # 80%+ → 對數增長，無上限
                    base = tns_weight * 60.0
                    excess = improvement_ratio - 0.8
                    # 對數增長：90% → 66, 95% → 71, 99% → 78, 99.9% → 85
                    milestone_reward += base + tns_weight * 20.0 * np.log1p(excess * 10)
            else:  # 惡化 - 大幅減輕懲罰
                degradation_ratio = abs(improvement_ratio)
                if degradation_ratio > 0.75:  # 惡化超過 75%
                    milestone_reward -= tns_weight * 5.0
                elif degradation_ratio > 0.50:  # 惡化超過 50%
                    milestone_reward -= tns_weight * 3.0
        else:  # 組合邏輯電路
            if global_tns_improvement > 0:
                milestone_reward += tns_weight * 8.0
            elif global_tns_improvement < 0:
                milestone_reward -= tns_weight * 1.0

        # --- Power 里程碑獎勵：分段 + 對數尾部設計 ---
        if abs(global_initial_power) > 1e-9:
            improvement_ratio = global_power_improvement / abs(global_initial_power)
            
            if improvement_ratio > 0:  # 有改善
                # 分段獎勵：前期線性，後期對數
                if improvement_ratio < 0.1:  # 0-10%
                    milestone_reward += power_weight * improvement_ratio * 80  # 最高 8
                elif improvement_ratio < 0.3:  # 10-30%
                    base = power_weight * 8.0
                    milestone_reward += base + power_weight * (improvement_ratio - 0.1) * 60  # 8 → 20
                elif improvement_ratio < 0.6:  # 30-60%
                    base = power_weight * 20.0
                    milestone_reward += base + power_weight * (improvement_ratio - 0.3) * 40  # 20 → 32
                else:  # 60%+ → 對數增長，無上限
                    base = power_weight * 32.0
                    excess = improvement_ratio - 0.6
                    # 對數增長：70% → 36, 80% → 40, 90% → 44
                    milestone_reward += base + power_weight * 15.0 * np.log1p(excess * 10)
            else:  # 惡化 - 大幅減輕懲罰
                degradation_ratio = abs(improvement_ratio)
                if degradation_ratio > 0.10:  # 惡化超過 10%
                    milestone_reward -= power_weight * 2.0

        # ========================================
        # 🏆 最終獎勵組合
        # ========================================
        total_reward = step_reward + milestone_reward

        # 🔧 獎勵範圍：保持負獎勵限制，但移除正獎勵上限
        # 負獎勵維持 -10.0，避免過度懲罰
        # 正獎勵不設上限，持續激勵優化
        reward = max(total_reward, -10.0)

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
