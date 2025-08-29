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
from datetime import datetime
from typing import Dict, List, Any

# Add path
sys.path.append('/root/ruan_workspace/ic_cad')

from config import RLConfig
from rl.environment import OptimizationEnvironment
from rl.ppo_agent import TwoDimensionalPPOAgent
from rl.training_controller import TrainingController, TrainingMonitor

logger = logging.getLogger(__name__)


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

        # Initialize agent with complete PPO parameters
        self.agent = TwoDimensionalPPOAgent(
            feature_dim=128,  # GNN embedding 維度 (128)
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
            max_grad_norm=config.max_grad_norm  # 梯度裁剪
        )
        
        # Initialize environment
        self.env = OptimizationEnvironment(config)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_tns_improvements = []
        self.episode_wns_improvements = []
        self.episode_power_improvements = []
        self.episode_success_rates = []
        
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
        
        # Training loop
        for episode in range(self.config.max_episodes):
            # Select case for this episode
            case_name = np.random.choice(case_names)
            
            # 顯示回合進度
            logger.info(f"🔄 開始回合 {episode + 1}/{self.config.max_episodes} - 案例: {case_name}")
            
            # Run episode
            episode_stats = self._run_episode(case_name, episode)
            
            # Store statistics
            self.episode_rewards.append(episode_stats['total_reward'])
            self.episode_lengths.append(episode_stats['episode_length'])
            self.episode_tns_improvements.append(episode_stats['tns_improvement'])
            self.episode_wns_improvements.append(episode_stats['wns_improvement'])
            self.episode_power_improvements.append(episode_stats['power_improvement'])
            self.episode_success_rates.append(episode_stats['success_rate'])
            
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
            
            # 記錄全域初始狀態 (第一回合的初始狀態)
            if self.global_initial_tns is None:
                self.global_initial_tns = episode_stats['initial_tns']
                self.global_initial_wns = episode_stats['initial_wns'] 
                self.global_initial_power = episode_stats['initial_power']
                logger.info(f"🎯 記錄最初電路狀態: TNS={self.global_initial_tns:.1f}ns, "
                          f"WNS={self.global_initial_wns:.1f}ns, Power={self.global_initial_power:.6f}W")
            
            # 計算與最初電路的絕對差距 (負值表示變差，正值表示改善)
            global_tns_improvement = self.global_initial_tns - final_tns
            global_wns_improvement = self.global_initial_wns - final_wns
            global_power_improvement = self.global_initial_power - final_power
            
            # 確保每個回合都顯示vs最初的基本比較
            logger.info(f"📊 回合 {episode + 1}/{self.config.max_episodes} - "
                       f"獎勵: {current_reward:.4f}, "
                       f"vs最初TNS改善: {global_tns_improvement:+.2f}ns, "
                       f"vs最初WNS改善: {global_wns_improvement:+.2f}ns")
            
            # 每5回合詳細顯示與最初電路的比較，確保用戶能看到持續的vs最初比較
            if (episode + 1) % 5 == 0 or episode < 10:
                logger.info(f"🔍 vs最初電路詳細比較 ({case_name}): "
                          f"TNS: {self.global_initial_tns:.1f}→{final_tns:.1f} "
                          f"({global_tns_improvement:+.1f}ns, {global_tns_improvement/abs(self.global_initial_tns)*100:+.1f}%), "
                          f"WNS: {self.global_initial_wns:.1f}→{final_wns:.1f} "
                          f"({global_wns_improvement:+.1f}ns, {global_wns_improvement/abs(self.global_initial_wns)*100:+.1f}%), "
                          f"Power: {self.global_initial_power:.6f}→{final_power:.6f} "
                          f"({global_power_improvement:+.6f}W, {global_power_improvement/self.global_initial_power*100:+.1f}%)")
            
            # 如果這是最後一個回合，一定要顯示最終的vs最初比較
            if episode + 1 == self.config.max_episodes:
                logger.info(f"🏁 最終vs最初電路比較 ({case_name}): "
                          f"TNS: {self.global_initial_tns:.1f}→{final_tns:.1f} "
                          f"({global_tns_improvement:+.1f}ns, {global_tns_improvement/abs(self.global_initial_tns)*100:+.1f}%), "
                          f"WNS: {self.global_initial_wns:.1f}→{final_wns:.1f} "
                          f"({global_wns_improvement:+.1f}ns, {global_wns_improvement/abs(self.global_initial_wns)*100:+.1f}%), "
                          f"Power: {self.global_initial_power:.6f}→{final_power:.6f} "
                          f"({global_power_improvement:+.6f}W, {global_power_improvement/self.global_initial_power*100:+.1f}%)")
            
            # Evaluation and saving
            if episode % self.config.save_interval == 0 and episode > 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                
                logger.info(f"💾 回合 {episode + 1}/{self.config.max_episodes} - 模型保存檢查點")
                logger.info(f"📈 平均獎勵: {avg_reward:.4f}, "
                          f"最新獎勵: {episode_stats['total_reward']:.4f}, "
                          f"成功率: {episode_stats['success_rate']:.2%}")
                
                # Ensure models directory exists
                models_dir = os.path.join(output_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)
                
                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_model_path = os.path.join(models_dir, 'best_model.pth')
                    self.agent.save_model(best_model_path)
                    logger.info(f"🏆 儲存最佳模型 - 平均獎勵: {best_avg_reward:.4f}")
                
                # Regular checkpoint
                checkpoint_path = os.path.join(models_dir, f'checkpoint_{episode + 1}.pth')
                self.agent.save_model(checkpoint_path)
                logger.info(f"✅ 儲存檢查點: checkpoint_{episode + 1}.pth")
                
                # Plot training curves
                self._plot_training_curves(output_dir)
        
        # Training completed
        models_dir = os.path.join(output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        final_model_path = os.path.join(models_dir, 'final_model.pth')
        self.agent.save_model(final_model_path)
        
        # Save training statistics
        self._save_training_stats(output_dir)
        
        training_time = datetime.now() - self.start_time
        logger.info(f"2D action space training completed - Total time: {training_time}")
        
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
        
        tns_improvement = initial_tns - final_tns
        wns_improvement = initial_wns - final_wns
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
                      f"Power: {initial_power:.6f}→{final_power:.6f} ({power_improvement:+.6f})")
        
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
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Episode rewards
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
            
            # Episode lengths
            axes[0, 1].plot(self.episode_lengths)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True)
            
            # Success rates
            axes[0, 2].plot(self.episode_success_rates)
            axes[0, 2].set_title('Action Success Rate')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Success Rate')
            axes[0, 2].grid(True)
            
            # TNS improvements
            axes[1, 0].plot(self.episode_tns_improvements)
            axes[1, 0].set_title('TNS Improvements')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('TNS Improvement')
            axes[1, 0].grid(True)
            
            # WNS improvements
            axes[1, 1].plot(self.episode_wns_improvements)
            axes[1, 1].set_title('WNS Improvements')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('WNS Improvement')
            axes[1, 1].grid(True)
            
            # Power improvements
            axes[1, 2].plot(self.episode_power_improvements)
            axes[1, 2].set_title('Power Improvements')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Power Improvement')
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, 'plots', 'training_curves_2d.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curves saved to: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to plot training curves: {e}")
    
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
