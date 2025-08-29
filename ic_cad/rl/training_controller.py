#!/usr/bin/env python3
"""
訓練控制器 - 提供早停和收斂判斷機制
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)

class TrainingController:
    """訓練控制器 - 管理訓練停止條件"""
    
    def __init__(self, 
                 patience: int = 100,           # 早停耐心值
                 min_improvement: float = 0.01,  # 最小改善閾值
                 convergence_window: int = 50,   # 收斂判斷窗口
                 convergence_threshold: float = 0.001):  # 收斂閾值
        """
        初始化訓練控制器
        
        Args:
            patience: 沒有改善時的等待回合數
            min_improvement: 認為有改善的最小閾值
            convergence_window: 檢查收斂的回合窗口
            convergence_threshold: 認為已收斂的獎勵變異閾值
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        
        # 早停相關
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.best_episode = 0
        
        # 收斂判斷
        self.reward_history = deque(maxlen=convergence_window)
        
        # 統計
        self.total_episodes = 0
        self.early_stop_triggered = False
        self.convergence_detected = False
        
    def should_stop(self, current_reward: float, episode: int) -> Dict[str, bool]:
        """
        判斷是否應該停止訓練
        
        Args:
            current_reward: 當前回合的獎勵
            episode: 當前回合數
            
        Returns:
            包含停止原因的字典
        """
        self.total_episodes = episode
        self.reward_history.append(current_reward)
        
        # 檢查是否有改善
        if current_reward > self.best_reward + self.min_improvement:
            self.best_reward = current_reward
            self.best_episode = episode
            self.episodes_without_improvement = 0
            logger.info(f"New best reward: {current_reward:.4f} at episode {episode}")
        else:
            self.episodes_without_improvement += 1
        
        # 早停檢查
        early_stop = self.episodes_without_improvement >= self.patience
        if early_stop and not self.early_stop_triggered:
            self.early_stop_triggered = True
            logger.info(f"Early stopping triggered at episode {episode} "
                       f"(no improvement for {self.episodes_without_improvement} episodes)")
        
        # 收斂檢查
        convergence = self._check_convergence()
        if convergence and not self.convergence_detected:
            self.convergence_detected = True
            logger.info(f"Convergence detected at episode {episode} "
                       f"(reward variance: {np.var(list(self.reward_history)):.6f})")
        
        return {
            'early_stop': early_stop,
            'convergence': convergence,
            'should_stop': early_stop or convergence
        }
    
    def _check_convergence(self) -> bool:
        """檢查是否已收斂"""
        if len(self.reward_history) < self.convergence_window:
            return False
        
        # 計算近期獎勵的變異數
        variance = np.var(list(self.reward_history))
        return variance < self.convergence_threshold
    
    def get_status(self) -> Dict:
        """獲取當前狀態"""
        return {
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward,
            'best_episode': self.best_episode,
            'episodes_without_improvement': self.episodes_without_improvement,
            'early_stop_triggered': self.early_stop_triggered,
            'convergence_detected': self.convergence_detected,
            'reward_variance': np.var(list(self.reward_history)) if len(self.reward_history) > 0 else 0.0
        }
    
    def reset(self):
        """重置控制器狀態"""
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.best_episode = 0
        self.reward_history.clear()
        self.total_episodes = 0
        self.early_stop_triggered = False
        self.convergence_detected = False


class TrainingMonitor:
    """訓練監控器 - 提供即時訓練狀態監控"""
    
    def __init__(self, log_interval: int = 10):
        """
        初始化訓練監控器
        
        Args:
            log_interval: 日誌輸出間隔
        """
        self.log_interval = log_interval
        self.episode_count = 0
        self.recent_rewards = deque(maxlen=100)  # 保存最近100個獎勵
        
    def update(self, episode: int, reward: float, episode_stats: Dict):
        """
        更新監控狀態
        
        Args:
            episode: 回合數
            reward: 獎勵
            episode_stats: 回合統計
        """
        self.episode_count = episode
        self.recent_rewards.append(reward)
        
        # 定期輸出狀態
        if episode % self.log_interval == 0:
            self._log_status(episode, reward, episode_stats)
    
    def _log_status(self, episode: int, reward: float, episode_stats: Dict):
        """輸出訓練狀態"""
        avg_reward = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0.0
        
        logger.info(f"Episode {episode:4d} | "
                   f"Reward: {reward:8.4f} | "
                   f"Avg(100): {avg_reward:8.4f} | "
                   f"TNS: {episode_stats.get('tns_improvement', 0):+.4f} | "
                   f"Power: {episode_stats.get('power_improvement', 0):+.6f} | "
                   f"Success: {episode_stats.get('success_rate', 0):.1%}")
    
    def get_training_summary(self) -> Dict:
        """獲取訓練摘要"""
        if not self.recent_rewards:
            return {}
        
        rewards = list(self.recent_rewards)
        return {
            'total_episodes': self.episode_count,
            'recent_avg_reward': np.mean(rewards),
            'recent_max_reward': np.max(rewards),
            'recent_min_reward': np.min(rewards),
            'reward_std': np.std(rewards),
            'improvement_trend': self._calculate_trend()
        }
    
    def _calculate_trend(self) -> str:
        """計算獎勵趨勢"""
        if len(self.recent_rewards) < 20:
            return "insufficient_data"
        
        rewards = list(self.recent_rewards)
        first_half = np.mean(rewards[:len(rewards)//2])
        second_half = np.mean(rewards[len(rewards)//2:])
        
        improvement = (second_half - first_half) / abs(first_half) if first_half != 0 else 0
        
        if improvement > 0.05:
            return "improving"
        elif improvement < -0.05:
            return "degrading"
        else:
            return "stable"
