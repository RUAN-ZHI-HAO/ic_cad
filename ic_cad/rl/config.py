"""RL Configuration
==========    # === PPO 參數 === (針對錯誤學    # === 智能停止參數 === (更積極的早停)
    early_stop_patience: int = 50       # 早停耐心值（減少）
    min_improvement: float = 0.005      # 認為有改善的最小閾值（降低）
    convergence_window: int = 30        # 收斂判斷窗口（縮短）
    convergence_threshold: float = 0.001  # 收斂閾值（獎勵變異）    learning_rate: float = 1e-5         # 學習率 (大幅降低)
    lr_actor: float = 1e-5              # Actor 網路學習率
    lr_critic: float = 2e-5             # Critic 網路學習率 (稍高)=====
強化學習系統的配置文件
"""

import os
from typing import Optional
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class RLConfig:
    """二維動作空間 RL 訓練配置"""
    
    # === 模型參數 ===
    gnn_feature_dim: int = 23           # GNN 節點輸入特徵維度 (22 基礎特徵 + 1 cell_id)
    gnn_embed_dim: int = 64         # GNN 嵌入維度 (配合新模型的 64 維輸出)
    cell_embedding_dim: int = 32        # Cell type embedding 維度
    effective_feature_dim: int = 54     # 實際處理維度 (22 基礎 + 32 embedding)
    dynamic_feature_dim: int = 12   # 動態特徵維度 (worst_slack, total_power, etc.)
    global_feature_dim: int = 9     # 全域特徵維度 (TNS, WNS, Power, etc.)
    hidden_dim: int = 256           # Policy/Value 網路隱藏層維度
    actor_hidden_dim: int = 256     # Actor 網路隱藏層維度
    critic_hidden_dim: int = 256    # Critic 網路隱藏層維度
    
    # === 二維動作空間參數 ===
    max_candidates: int = 20        # 最大候選數量
    max_replacements: int = 84      # 最大替換選項數 (來自最大 cell group 大小)
    candidate_selection_top_p: int = 15  # Top-P worst slack candidates (更聚焦)
    candidate_selection_top_h: int = 5   # Top-H high power candidates (降低)
    
    # === PPO 參數 === (針對穩定和有效的 PPO 訓練優化)
    learning_rate: float = 1e-4        # 學習率 (降低避免過度更新)
    lr_actor: float = 1e-4             # Actor 網路學習率 (降低)
    lr_critic: float = 2e-4            # Critic 網路學習率 (稍高幫助價值估計)
    clip_range: float = 0.1            # PPO 裁剪範圍 (降低避免太激進)
    eps_clip: float = 0.1              # PPO epsilon 裁剪 (降低)
    gamma: float = 0.99                # 折扣因子 (標準設定)
    gae_lambda: float = 0.95           # GAE lambda 參數
    entropy_coef: float = 0.05         # 熵係數 (增加探索能力)
    value_coef: float = 0.5            # 價值函數係數
    target_kl: float = 0.02            # 目標 KL 散度 (放寬避免頻繁早停)
    max_grad_norm: float = 0.5         # 梯度裁剪 (標準設定)
    ppo_epochs: int = 3                # PPO 更新次數 (減少避免過擬合)
    batch_size: int = 64               # 批次大小
    buffer_size: int = 2048            # 經驗緩衝區大小
    buffer_capacity: int = 2048        # 緩衝區容量（別名）
    
    # === 訓練參數 === (平衡效率與效果)
    max_episodes: int = 2000        # 最大訓練回合數 (增加)
    max_steps_per_episode: int = 25 # 每回合最大步數 (折衷方案)
    update_interval: int = 256      # 更新間隔
    save_interval: int = 50         # 模型保存間隔 (更頻繁保存)
    eval_interval: int = 25         # 評估間隔 (更頻繁評估)
    
    # === 智能停止參數 === (平衡效率與完整性)
    early_stop_patience: int = 75   # 早停耐心值（適中）
    min_improvement: float = 0.008  # 認為有改善的最小閾值 (適中)
    convergence_window: int = 40    # 收斂判斷窗口 (適中)
    convergence_threshold: float = 0.0005  # 收斂閾值（更嚴格的收斂）
    
    # === 環境參數 ===
    benchmark_cases: List[str] = None  # 測試案例列表
    target_frequency: float = 1e9      # 目標頻率 (Hz)
    
    # === 獎勵權重 === (更平衡的長期訓練設定)
    reward_weight_tns: float = 3.0     # TNS 獎勵權重 (大幅降低避免過於激進)
    reward_weight_wns: float = 2.0     # WNS 獎勵權重 (降低)
    reward_weight_power: float = 0.5   # Power 獎勵權重 (降低)
    fail_penalty: float = 0.1          # 失敗懲罰 (更寬鬆)
    tns_goal_threshold: float = -0.1   # TNS 目標閾值 (更實際)
    wns_goal_threshold: float = -0.1   # WNS 目標閾值 (更實際)
    goal_bonus: float = 1.0            # 目標達成獎勵 (適中)
    
    # === 動態權重配置 ===
    use_dynamic_weights: bool = False     # 是否使用動態權重
    dynamic_tns_weight: float = 1.0       # 動態 TNS 權重
    dynamic_power_weight: float = 1.0     # 動態 Power 權重
    normalize_weights: bool = True        # 是否正規化權重
    
    # === 設備配置 ===
    device: str = "auto"  # "auto", "cuda", "cpu" - auto 會自動選擇最佳設備
    force_cpu: bool = False  # 強制使用 CPU（即使有 GPU）
    
    # === 檔案路徑 ===
    # gnn_model_path 設為 None，讓 load_encoder 自動從 meta 中讀取
    gnn_model_path: Optional[str] = None  
    gnn_meta_path: str = "/root/ruan_workspace/ic_cad/gnn/encoder_meta.json"
    benchmark_dir: str = "/root/ruan_workspace/ic_cad/open_ic_design"
    cell_groups_path: str = "/root/ruan_workspace/ic_cad/gnn/cell_groups.json"
    
    # === OpenROAD 配置 ===
    openroad_work_dir: str = "/tmp/openroad_work"
    openroad_pdk_root: str = "~/solution/testcases/ASAP7"
    openroad_design_root: str = "/root/solution/testcases"
    auto_repair_each_step: bool = True
    
    # === 動作庫配置 === (已廢棄，現在使用動態元件替換)
    demo_target_cell: str = "_1_"
    demo_new_master: str = "NAND2xp33_ASAP7_75t_L"
    
    # === 輸出路徑 ===
    output_dir: str = "/root/ruan_workspace/ic_cad/rl/outputs"
    model_save_dir: str = "/root/ruan_workspace/ic_cad/rl/models"
    log_dir: str = "/root/ruan_workspace/ic_cad/rl/logs"
    
    def __post_init__(self):
        """初始化後處理"""
        if self.benchmark_cases is None:
            # self.benchmark_cases = ['c17', 'c432', 'c499', 'c880', 'c1355', 'c1908', 'c2670']
            self.benchmark_cases = ['s1488']
        
        # 設備配置
        import torch
        if self.force_cpu:
            self.device = "cpu"
        elif self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            print("⚠️  警告：請求使用 CUDA 但 GPU 不可用，回退到 CPU")
            self.device = "cpu"
        
        print(f"🖥️  使用設備: {self.device.upper()}")
        if self.device == "cuda":
            print(f"🚀 GPU 型號: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 創建輸出目錄
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
@dataclass
class InferenceConfig:
    """推論配置"""
    
    # === 設備配置 ===
    device: str = "auto"  # "auto", "cuda", "cpu"
    force_cpu: bool = False  # 強制使用 CPU
    
    # === 模型路徑 ===
    actor_model_path: str = "/root/ruan_workspace/ic_cad/rl/models/best_actor.pth"
    # gnn_model_path 設為 None，讓 load_encoder 自動從 meta 中讀取
    gnn_model_path: Optional[str] = None
    gnn_meta_path: str = "/root/ruan_workspace/ic_cad/gnn/encoder_meta.json"
    
    # === 推論參數 ===
    max_actions: int = 10      # 最大動作數量
    greedy: bool = False       # 是否使用貪婪策略
    temperature: float = 1.0   # 溫度參數 (用於動作採樣)
    
    # === 輸出設定 ===
    save_results: bool = True  # 是否保存結果
    output_dir: str = "/root/ruan_workspace/ic_cad/rl/inference_results"
    
    def __post_init__(self):
        """初始化後處理"""
        # 設備配置
        import torch
        if self.force_cpu:
            self.device = "cpu"
        elif self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            print("⚠️  警告：請求使用 CUDA 但 GPU 不可用，回退到 CPU")
            self.device = "cpu"
        
        print(f"🖥️  推論設備: {self.device.upper()}")
        
        os.makedirs(self.output_dir, exist_ok=True)

# 預設配置實例
default_rl_config = RLConfig()
default_inference_config = InferenceConfig()
