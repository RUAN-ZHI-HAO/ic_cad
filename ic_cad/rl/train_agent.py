"""
2D Action Space Training Manager for IC CAD Optimization

This module provides training functionality for 2D action space PPO agents.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from typing import Dict, List, Any

# Add path
sys.path.append('/root/ruan_workspace/ic_cad')

from config import RLConfig
from rl.environment import OptimizationEnvironment
from rl.ppo_agent import TwoDimensionalPPOAgent
from rl.training_controller import TrainingController, TrainingMonitor

logger = logging.getLogger(__name__)

def pct_str(impr: float, base: float, use_abs_base: bool = True) -> str:
    """回傳像 +12.3% 或 N/A 的字串；base 為 0（或太小）時回 N/A。"""
    EPS = 1e-12  # 避免浮點誤差導致誤判為 0
    denom = abs(base) if use_abs_base else base
    if denom is None or math.isclose(denom, 0.0, abs_tol=EPS):
        return "N/A"
    return f"{(impr / denom) * 100:+.1f}%"

class TwoDimensionalTrainingManager:
    """Training manager for 2D action space RL agents"""
    
    def __init__(self, config: RLConfig):
        """
        Initialize training manager
        
        Args:
            config: RL configuration object
        """
        self.config = config
        self.start_time = None

        # Initialize environment first to read GNN dimensions
        self.env = OptimizationEnvironment(config)
        
        # 動態讀取 GNN embedding 維度，而非硬編碼
        gnn_embed_dim = self.env.gnn_embed_dim  # 從環境中讀取實際維度
        logger.info(f"🔧 使用動態 GNN 嵌入維度: {gnn_embed_dim}")
        
        # Initialize agent with complete PPO parameters
        self.agent = TwoDimensionalPPOAgent(
            feature_dim=gnn_embed_dim,  # 動態 GNN embedding 維度
            hidden_dim=config.hidden_dim,  # 從 config 讀取
            max_candidates=config.max_candidates,
            max_replacements=config.max_replacements,
            lr=config.lr_actor,  # Actor 學習率
            critic_lr=config.lr_critic,  # Critic 學習率
            gamma=config.gamma,
            eps_clip=config.eps_clip,
            ppo_epochs=config.ppo_epochs,  # PPO 更新次數
            mini_batch_size=config.batch_size,  # Mini-batch 大小
            entropy_coef=config.entropy_coef,  # 熵係數
            value_coef=config.value_coef,  # 價值函數係數
            target_kl=config.target_kl,  # 目標 KL 散度
            max_grad_norm=config.max_grad_norm,  # 梯度裁剪
            device=config.device  # 傳遞設備配置
        )
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_tns_improvements = []
        self.episode_wns_improvements = []
        self.episode_power_improvements = []
        self.episode_success_rates = []
        self.episode_final_tns = []
        self.episode_final_wns = []
        self.episode_final_power = []
        
        # Global initial state tracking (第一次重置時的電路狀態)
        self.global_initial_tns = None
        self.global_initial_wns = None
        self.global_initial_power = None
        
        logger.info("TwoDimensionalTrainingManager initialized")
        logger.info(f"Max candidates: {config.max_candidates}, Max replacements: {config.max_replacements}")
    
    def train(self, case_names: List[str], output_dir: str) -> Dict[str, Any]:
        """
        Train the 2D action space agent
        
        Args:
            case_names: List of test case names
            output_dir: Output directory for results
            
        Returns:
            Training results dictionary
        """
        logger.info(f"🚀 開始 2D 動作空間訓練 - 真正的 PPO 實現")
        logger.info(f"📋 訓練案例: {case_names}")
        
        # 多 case 訓練警告
        if len(case_names) > 1:
            logger.warning("⚠️  多案例訓練模式已啟用")
            logger.warning("⚠️  注意：OpenROAD 在切換案例時可能出現狀態管理問題")
            logger.warning("⚠️  建議：如果遇到 Segmentation Fault，請分別訓練每個案例")
            logger.warning("⚠️  或使用獨立的環境實例來避免狀態衝突")
        
        logger.info(f"🔢 總回合數: {self.config.max_episodes}")
        logger.info(f"💾 保存間隔: 每 {self.config.save_interval} 回合")
        logger.info(f"📁 輸出目錄: {output_dir}")
        logger.info(f"🔧 PPO 參數: epochs={self.config.ppo_epochs}, "
                   f"clip={self.config.eps_clip}, lr={self.config.lr_actor}, "
                   f"entropy_coef={self.config.entropy_coef}")
        logger.info(f"📊 更新策略: 每 {self.config.update_interval} 步或緩衝區滿時更新")
        logger.info("=" * 80)
        self.start_time = datetime.now()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        best_avg_reward = float('-inf')
        best_episode_info = None  # 記錄最佳 episode 的詳細資訊 (基於獎勵)
        
        # 追蹤最佳 TNS 和 Power 的 episode
        best_tns = float('-inf')
        best_tns_episode_info = None
        best_power = float('inf')
        best_power_episode_info = None
        
        # Training loop
        for episode in range(self.config.max_episodes):
            # Select case for this episode
            # 🔒 安全策略：如果只有一個 case，直接使用；多個 case 時輪流訓練
            # 注意：多 case 訓練可能因 OpenROAD 狀態管理問題導致崩潰
            if len(case_names) == 1:
                case_name = case_names[0]
                cycle_info = ""
            else:
                case_name = case_names[episode % len(case_names)]
                cycle_info = f" ({(episode % len(case_names)) + 1}/{len(case_names)} in cycle {episode // len(case_names) + 1})"
            
            # 顯示回合進度
            logger.info(f"🔄 開始回合 {episode + 1}/{self.config.max_episodes} - 案例: {case_name}{cycle_info}")
            
            # Run episode
            episode_stats = self._run_episode(case_name, episode)
            
            # Store statistics
            self.episode_rewards.append(episode_stats['total_reward'])
            self.episode_lengths.append(episode_stats['episode_length'])
            
            # 記錄相對於全域初始狀態的改善（用於繪圖）
            if self.global_initial_tns is not None:
                global_tns_improvement = episode_stats['final_tns'] - self.global_initial_tns
                global_wns_improvement = episode_stats['final_wns'] - self.global_initial_wns
                # Power 改善 = 初始值 - 最終值（降低是好的，所以反過來）
                global_power_improvement = self.global_initial_power - episode_stats['final_power']
            else:
                # 第一個 episode，使用 episode 內部改善
                global_tns_improvement = episode_stats['tns_improvement']
                global_wns_improvement = episode_stats['wns_improvement']
                global_power_improvement = episode_stats['power_improvement']
            
            self.episode_tns_improvements.append(global_tns_improvement)
            self.episode_wns_improvements.append(global_wns_improvement)
            self.episode_power_improvements.append(global_power_improvement)
            self.episode_success_rates.append(episode_stats['success_rate'])
            self.episode_final_tns.append(episode_stats['final_tns'])
            self.episode_final_wns.append(episode_stats['final_wns'])
            self.episode_final_power.append(episode_stats['final_power'])
            
            # Update agent - 改用更合適的更新策略
            buffer_steps = len(self.agent.buffer['rewards']) if 'rewards' in self.agent.buffer else 0
            
            # 每 update_interval 步或緩衝區滿時更新（更符合 PPO 標準）
            should_update = (
                buffer_steps >= self.config.update_interval or  # 定期更新
                buffer_steps >= self.config.buffer_size or      # 緩衝區滿
                (buffer_steps >= self.config.batch_size and episode % 5 == 0)  # 小批量定期更新
            )
            
            if should_update and buffer_steps > 0:
                update_stats = self.agent.update()
                
                if update_stats:  # 顯示更新統計
                    logger.info(f"🔧 回合 {episode + 1} - PPO 更新 (緩衝: {buffer_steps}步): "
                              f"策略損失: {update_stats['policy_loss']:.4f}, "
                              f"價值損失: {update_stats['value_loss']:.4f}, "
                              f"熵: {update_stats['entropy']:.4f}, "
                              f"KL散度: {update_stats.get('approx_kl', 0):.4f}, "
                              f"裁剪率: {update_stats.get('clip_fraction', 0):.2%}, "
                              f"執行epoch: {update_stats.get('epochs_ran', 0)}")
            else:
                if episode % 50 == 0:  # 定期顯示緩衝區狀態
                    logger.info(f"📊 回合 {episode + 1} - 緩衝區累積: {buffer_steps}/{self.config.update_interval}步")
            
            # 每回合顯示基本進度 - 包含與最初電路的絕對差距
            current_reward = episode_stats['total_reward']
            final_tns = episode_stats['final_tns']
            final_wns = episode_stats['final_wns']
            final_power = episode_stats['final_power']
            
            # 檢查並更新最佳 TNS 紀錄
            if final_tns > best_tns:
                best_tns = final_tns
                best_tns_episode_info = {
                    'episode': episode + 1,
                    'case_name': case_name,
                    'total_reward': current_reward,
                    'final_tns': final_tns,
                    'final_wns': final_wns,
                    'final_power': final_power,
                    'tns_improvement': global_tns_improvement,
                    'power_improvement': global_power_improvement
                }
                
            # 檢查並更新最佳 Power 紀錄
            if final_power < best_power:
                best_power = final_power
                best_power_episode_info = {
                    'episode': episode + 1,
                    'case_name': case_name,
                    'total_reward': current_reward,
                    'final_tns': final_tns,
                    'final_wns': final_wns,
                    'final_power': final_power,
                    'tns_improvement': global_tns_improvement,
                    'power_improvement': global_power_improvement
                }
            
            # 記錄全域初始狀態 (第一回合的初始狀態)
            if self.global_initial_tns is None:
                self.global_initial_tns = episode_stats['initial_tns']
                self.global_initial_wns = episode_stats['initial_wns'] 
                self.global_initial_power = episode_stats['initial_power']
                logger.info(f"🎯 記錄最初電路狀態: TNS={self.global_initial_tns:.1f}ns, "
                          f"WNS={self.global_initial_wns:.1f}ns, Power={self.global_initial_power:.6f}W")
            
            # 計算與最初電路的絕對差距 (負值表示變差，正值表示改善)
            global_tns_improvement = final_tns - self.global_initial_tns
            global_wns_improvement = final_wns - self.global_initial_wns
            global_power_improvement = final_power - self.global_initial_power
            
            # 確保每個回合都顯示vs最初的基本比較
            logger.info(f"📊 回合 {episode + 1}/{self.config.max_episodes} - "
                       f"獎勵: {current_reward:.4f}, "
                       f"vs最初TNS改善: {global_tns_improvement:+.2f}ns, "
                       f"vs最初WNS改善: {global_wns_improvement:+.2f}ns")
            
            # 每5回合詳細顯示與最初電路的比較，確保用戶能看到持續的vs最初比較
            # if (episode + 1) % 5 == 0 or episode < 10:
            # 防止除零錯誤
            tns_pct = (global_tns_improvement/abs(self.global_initial_tns)*100) if self.global_initial_tns != 0 else 0.0
            wns_pct = (global_wns_improvement/abs(self.global_initial_wns)*100) if self.global_initial_wns != 0 else 0.0
            power_pct = (global_power_improvement/self.global_initial_power*100) if self.global_initial_power != 0 else 0.0
            
            logger.info(f"🔍 vs最初電路詳細比較 ({case_name}): "
                        f"TNS: {self.global_initial_tns:.1f}→{final_tns:.1f} "
                        f"({global_tns_improvement:+.1f}ns, {tns_pct:+.1f}%), "
                        f"WNS: {self.global_initial_wns:.1f}→{final_wns:.1f} "
                        f"({global_wns_improvement:+.1f}ns, {wns_pct:+.1f}%), "
                        f"Power: {self.global_initial_power:.6f}→{final_power:.6f} "
                        f"({global_power_improvement:+.6f}W, {power_pct:+.1f}%)")

            # 如果這是最後一個回合，一定要顯示最終的vs最初比較
            if episode + 1 == self.config.max_episodes:
                tns_pct   = pct_str(global_tns_improvement,   self.global_initial_tns,  use_abs_base=True)
                wns_pct   = pct_str(global_wns_improvement,   self.global_initial_wns,  use_abs_base=True)
                power_pct = pct_str(global_power_improvement, self.global_initial_power, use_abs_base=False)

                logger.info(
                    f"🏁 最終vs最初電路比較 ({case_name}): "
                    f"TNS: {self.global_initial_tns:.1f}→{final_tns:.1f} "
                    f"({global_tns_improvement:+.1f}ns, {tns_pct}), "
                    f"WNS: {self.global_initial_wns:.1f}→{final_wns:.1f} "
                    f"({global_wns_improvement:+.1f}ns, {wns_pct}), "
                    f"Power: {self.global_initial_power:.6f}→{final_power:.6f} "
                    f"({global_power_improvement:+.6f}W, {power_pct})"
                )
            
            # 每回合檢查是否為最佳成效（靜默更新，不顯示）
            avg_reward = np.mean(self.episode_rewards[-min(100, len(self.episode_rewards)):])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                # Ensure models directory exists
                models_dir = os.path.join(output_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)
                
                best_model_path = os.path.join(output_dir, 'models', 'best_model.pth')
                self.agent.save_model(best_model_path)
                
                # 記錄最佳 episode 資訊
                best_episode_info = {
                    'episode': episode + 1,
                    'case_name': case_name,
                    'avg_reward': avg_reward,
                    'total_reward': episode_stats['total_reward'],
                    'tns_improvement': episode_stats['tns_improvement'],
                    'wns_improvement': episode_stats['wns_improvement'],
                    'power_improvement': episode_stats['power_improvement'],
                    'success_rate': episode_stats['success_rate'],
                    'initial_tns': episode_stats['initial_tns'],
                    'final_tns': episode_stats['final_tns'],
                    'initial_wns': episode_stats['initial_wns'],
                    'final_wns': episode_stats['final_wns'],
                    'initial_power': episode_stats['initial_power'],
                    'final_power': episode_stats['final_power']
                }
            
            # Evaluation and saving
            if episode % self.config.save_interval == 0 and episode > 0:
                logger.info(f"💾 回合 {episode + 1}/{self.config.max_episodes} - 模型保存檢查點")
                logger.info(f"📈 平均獎勵: {avg_reward:.4f}, "
                          f"最新獎勵: {episode_stats['total_reward']:.4f}, "
                          f"成功率: {episode_stats['success_rate']:.2%}")
                
                # Ensure models directory exists
                models_dir = os.path.join(output_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)
                
                # best_model 已經在上面的每回合檢查中保存了
                if False:  # 不再需要這段，已經移到上面
                    pass
                
                # Regular checkpoint
                checkpoint_path = os.path.join(models_dir, f'checkpoint_{episode + 1}.pth')
                self.agent.save_model(checkpoint_path)
                logger.info(f"✅ 儲存檢查點: checkpoint_{episode + 1}.pth")
                
                # Plot training curves
                self._plot_training_curves(output_dir)
                self._plot_absolute_metrics(output_dir)
        
        # Training completed
        models_dir = os.path.join(output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        final_model_path = os.path.join(models_dir, 'final_model.pth')
        self.agent.save_model(final_model_path)
        
        # Save training statistics
        self._save_training_stats(output_dir)
        
        training_time = datetime.now() - self.start_time
        logger.info("=" * 80)
        logger.info(f"✅ 2D 動作空間訓練完成 - 總時間: {training_time}")
        logger.info("=" * 80)
        
        # 顯示最佳成效紀錄 (基於獎勵)
        if best_episode_info:
            logger.info(f"🏆 最佳成效紀錄 (基於平均獎勵):")
            logger.info(f"  回合: {best_episode_info['episode']}/{self.config.max_episodes}")
            logger.info(f"  案例: {best_episode_info['case_name']}")
            logger.info(f"  平均獎勵: {best_episode_info['avg_reward']:.4f}")
            logger.info(f"  總獎勵: {best_episode_info['total_reward']:.4f}")
            logger.info(f"  成功率: {best_episode_info['success_rate']:.2%}")
            logger.info(f"  TNS: {best_episode_info['initial_tns']:.4f} → {best_episode_info['final_tns']:.4f} ({best_episode_info['tns_improvement']:+.4f} ns)")
            logger.info(f"  WNS: {best_episode_info['initial_wns']:.4f} → {best_episode_info['final_wns']:.4f} ({best_episode_info['wns_improvement']:+.4f} ns)")
            logger.info(f"  Power: {best_episode_info['initial_power']:.6f} → {best_episode_info['final_power']:.6f} ({best_episode_info['power_improvement']:+.6f} W)")
            logger.info("=" * 80)
            
        # 顯示最佳 TNS 紀錄
        if best_tns_episode_info:
            logger.info(f"⚡ TNS 優化最多的時候:")
            logger.info(f"  回合: {best_tns_episode_info['episode']}")
            logger.info(f"  TNS: {best_tns_episode_info['final_tns']:.4f} ns")
            logger.info(f"  Power: {best_tns_episode_info['final_power']:.6f} W")
            logger.info("=" * 80)

        # 顯示最佳 Power 紀錄
        if best_power_episode_info:
            logger.info(f"🔋 Power 優化最多的時候:")
            logger.info(f"  回合: {best_power_episode_info['episode']}")
            logger.info(f"  TNS: {best_power_episode_info['final_tns']:.4f} ns")
            logger.info(f"  Power: {best_power_episode_info['final_power']:.6f} W")
            logger.info("=" * 80)
        
        logger.info(f"📊 訓練統計:")
        logger.info(f"  平均獎勵: {np.mean(self.episode_rewards):.4f}")
        logger.info(f"  最高獎勵: {np.max(self.episode_rewards):.4f}")
        logger.info(f"  平均 TNS 改善: {np.mean(self.episode_tns_improvements):+.4f} ns")
        logger.info(f"  平均 WNS 改善: {np.mean(self.episode_wns_improvements):+.4f} ns")
        logger.info(f"  平均 Power 改善: {np.mean(self.episode_power_improvements):+.6f} W")
        logger.info(f"  平均成功率: {np.mean(self.episode_success_rates):.2%}")
        logger.info("=" * 80)
        
        results = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'tns_improvements': self.episode_tns_improvements,
            'wns_improvements': self.episode_wns_improvements,
            'power_improvements': self.episode_power_improvements,
            'success_rates': self.episode_success_rates
        }
        
        return results
    
    def _run_episode(self, case_name: str, episode: int) -> Dict[str, float]:
        """
        Run a single 2D action space episode using real environment
        
        Args:
            case_name: Test case name
            episode: Episode number
            
        Returns:
            Episode statistics
        """
        # Reset environment for new episode
        state = self.env.reset(case_name)
        
        total_reward = 0.0
        step_count = 0
        successful_actions = 0
        max_steps = self.config.max_steps_per_episode
        
        # Record initial metrics
        initial_tns = state.current_tns
        initial_wns = state.current_wns
        initial_power = state.current_power
        
        # 🔍 顯示訓練回合的進度配置（與推論保持一致）
        if episode % 50 == 0:  # 每 50 回合顯示一次
            logger.info(f"🎯 回合 {episode + 1} 進度配置 - max_steps: {max_steps}, "
                       f"RL agent 將透過 global_features[3] 和 [6] 接收進度資訊")
        
        for step in range(max_steps):
            # Get action from agent
            action, info = self.agent.get_action(state, deterministic=False)
            
            # Execute action in real environment
            next_state, reward, done, env_info = self.env.step(action)
            
            # Store reward for the last action
            self.agent.store_reward(reward, done)
            
            # Update statistics
            total_reward += reward
            step_count += 1
            if env_info.get('success', False):
                successful_actions += 1
            
            # Update state
            state = next_state
            
            if done:
                break
        
        # Calculate improvements from final state
        final_tns = state.current_tns
        final_wns = state.current_wns
        final_power = state.current_power
        
        tns_improvement = final_tns - initial_tns
        wns_improvement = final_wns - initial_wns
        # Power 改善 = 初始值 - 最終值（降低是正改善）
        power_improvement = initial_power - final_power
        success_rate = successful_actions / step_count if step_count > 0 else 0.0
        
        # Log episode information
        if episode % 10 == 0 or episode < 20:  # 前20回合每回合顯示，之後每10回合顯示
            logger.info(f"✅ 回合 {episode + 1} 完成 ({case_name}) - "
                      f"獎勵: {total_reward:.4f}, "
                      f"步數: {step_count}, "
                      f"成功率: {success_rate:.2%}, "
                      f"TNS: {initial_tns:.4f}→{final_tns:.4f} ({tns_improvement:+.4f}), "
                      f"WNS: {initial_wns:.4f}→{final_wns:.4f} ({wns_improvement:+.4f}), "
                      f"Power: {initial_power:.6f}→{final_power:.6f} (改善: {power_improvement:+.6f})")
        
        return {
            'total_reward': total_reward,
            'episode_length': step_count,
            'tns_improvement': tns_improvement,
            'wns_improvement': wns_improvement,
            'power_improvement': power_improvement,
            'success_rate': success_rate,
            'initial_tns': initial_tns,
            'final_tns': final_tns,
            'initial_wns': initial_wns,
            'final_wns': final_wns,
            'initial_power': initial_power,
            'final_power': final_power
        }
    
    def _plot_training_curves(self, output_dir: str):
        """Plot 2D action space training curves"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Episode rewards
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
            
            # TNS improvements
            tns_data = np.array(self.episode_tns_improvements)
            axes[0, 1].plot(tns_data)
            axes[0, 1].set_title('TNS Improvements (vs Initial)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('TNS Improvement (ns)')
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
            axes[0, 1].grid(True)
            
            # 自動檢測是否需要對數刻度 (當數據範圍過大時)
            if len(tns_data) > 0:
                tns_range = np.max(tns_data) - np.min(tns_data)
                if tns_range > 10000:  # 範圍超過 10000 ns
                    axes[0, 1].set_yscale('symlog', linthresh=100)
                    axes[0, 1].set_title('TNS Improvements (vs Initial) [SymLog Scale]')
            
            # WNS improvements
            wns_data = np.array(self.episode_wns_improvements)
            axes[1, 0].plot(wns_data)
            axes[1, 0].set_title('WNS Improvements (vs Initial)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('WNS Improvement (ns)')
            axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
            axes[1, 0].grid(True)
            
            if len(wns_data) > 0:
                wns_range = np.max(wns_data) - np.min(wns_data)
                if wns_range > 1000:  # 範圍超過 1000 ns
                    axes[1, 0].set_yscale('symlog', linthresh=10)
                    axes[1, 0].set_title('WNS Improvements (vs Initial) [SymLog Scale]')
            
            # Power improvements
            axes[1, 1].plot(self.episode_power_improvements)
            axes[1, 1].set_title('Power Improvements (vs Initial)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Power Improvement (W)')
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, 'plots', 'training_curves_2d.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curves saved to: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to plot training curves: {e}")

    def _plot_absolute_metrics(self, output_dir: str):
        """Plot absolute metrics (TNS, WNS, Power)"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))
            
            # Final TNS
            tns_data = np.array(self.episode_final_tns)
            axes[0].plot(tns_data)
            axes[0].set_title('Final TNS per Episode')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('TNS (ns)')
            axes[0].grid(True)
            
            if len(tns_data) > 0:
                tns_range = np.max(tns_data) - np.min(tns_data)
                if tns_range > 10000 or np.min(tns_data) < -10000:
                    axes[0].set_yscale('symlog', linthresh=100)
                    axes[0].set_title('Final TNS per Episode [SymLog Scale]')

            # Final WNS
            wns_data = np.array(self.episode_final_wns)
            axes[1].plot(wns_data)
            axes[1].set_title('Final WNS per Episode')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('WNS (ns)')
            axes[1].grid(True)
            
            if len(wns_data) > 0:
                wns_range = np.max(wns_data) - np.min(wns_data)
                if wns_range > 1000 or np.min(wns_data) < -1000:
                    axes[1].set_yscale('symlog', linthresh=10)
                    axes[1].set_title('Final WNS per Episode [SymLog Scale]')

            # Final Power
            axes[2].plot(self.episode_final_power)
            axes[2].set_title('Final Power per Episode')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Power (W)')
            axes[2].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, 'plots', 'training_metrics_absolute.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Absolute metrics curves saved to: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to plot absolute metrics: {e}")
    
    def _save_training_stats(self, output_dir: str):
        """Save training statistics"""
        stats = {
            'config': {
                'max_episodes': self.config.max_episodes,
                'max_steps_per_episode': self.config.max_steps_per_episode,
                'max_candidates': self.config.max_candidates,
                'max_replacements': self.config.max_replacements,
                'lr_actor': self.config.lr_actor,
                'lr_critic': self.config.lr_critic,
                'gamma': self.config.gamma,
                'gae_lambda': self.config.gae_lambda,
                'eps_clip': self.config.eps_clip,
                'entropy_coef': self.config.entropy_coef,
                'batch_size': self.config.batch_size
            },
            'training_stats': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'tns_improvements': self.episode_tns_improvements,
                'wns_improvements': self.episode_wns_improvements,
                'power_improvements': self.episode_power_improvements,
                'success_rates': self.episode_success_rates,
                'final_tns': self.episode_final_tns,
                'final_wns': self.episode_final_wns,
                'final_power': self.episode_final_power,
                'agent_stats': self.agent.training_stats
            },
            'summary': {
                'total_episodes': len(self.episode_rewards),
                'average_reward': np.mean(self.episode_rewards),
                'best_reward': np.max(self.episode_rewards),
                'average_length': np.mean(self.episode_lengths),
                'average_tns_improvement': np.mean(self.episode_tns_improvements),
                'average_wns_improvement': np.mean(self.episode_wns_improvements),
                'average_power_improvement': np.mean(self.episode_power_improvements),
                'average_success_rate': np.mean(self.episode_success_rates)
            }
        }
        
        stats_path = os.path.join(output_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Training statistics saved to: {stats_path}")


def main():
    """Main function for training"""
    parser = argparse.ArgumentParser(description='Train 2D Action Space IC CAD RL Agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Critic learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--cases', nargs='+', default=['c17'], help='Test cases')
    parser.add_argument('--output_dir', type=str, 
                       default='/root/ruan_workspace/ic_cad/rl/training_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create configuration
    config = RLConfig()
    
    # Override parameters
    config.max_episodes = args.episodes
    config.lr_actor = args.lr_actor
    config.lr_critic = args.lr_critic
    config.batch_size = args.batch_size
    
    logger.info(f"Starting 2D action space training")
    logger.info(f"Config: episodes={config.max_episodes}, cases={args.cases}")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create training manager
        trainer = TwoDimensionalTrainingManager(config)
        
        # Execute training
        training_stats = trainer.train(args.cases, args.output_dir)
        
        logger.info("2D action space training completed!")
        logger.info(f"Average reward: {np.mean(training_stats['episode_rewards']):.4f}")
        logger.info(f"Average success rate: {np.mean(training_stats['success_rates']):.2%}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
