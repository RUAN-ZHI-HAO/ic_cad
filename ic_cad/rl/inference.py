"""Inference Script for 2D Action Space RL Agent
===============================================
使用訓練好的二維動作空間 RL 智能體進行推論
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加路徑
sys.path.append('/root/ruan_workspace/ic_cad/rl')
sys.path.append('/root/ruan_workspace/ic_cad/gnn')

from config import InferenceConfig, RLConfig
from environment import OptimizationEnvironment
from ppo_agent import TwoDimensionalPPOAgent
from utils_openroad import OptimizationAction

class TwoDimensionalInferenceEngine:
    """二維動作空間推論引擎"""
    
    def __init__(self, inference_config: InferenceConfig, rl_config: RLConfig):
        """
        初始化推論引擎
        
        Args:
            inference_config: 推論配置
            rl_config: RL 配置
        """
        self.inference_config = inference_config
        self.rl_config = rl_config
        
        # 建立環境
        logger.info("建立二維動作空間環境...")
        self.env = OptimizationEnvironment(self.rl_config)
        
        # 建立智能體
        logger.info("建立二維動作空間智能體...")
        self.agent = TwoDimensionalPPOAgent(
            feature_dim=self.rl_config.gnn_embed_dim,
            hidden_dim=self.rl_config.hidden_dim,
            max_candidates=self.rl_config.max_candidates,
            max_replacements=self.rl_config.max_replacements,
            device=inference_config.device  # 傳遞設備配置
        )
        
        # 結果記錄
        self.optimization_results = []
        
        logger.info("二維動作空間推論引擎初始化完成")
    
    def load_model(self, model_path: str):
        """載入訓練好的模型"""
        try:
            if os.path.exists(model_path):
                self.agent.load_model(model_path)
                logger.info(f"成功載入模型: {model_path}")
            else:
                logger.warning(f"模型檔案不存在: {model_path}")
                logger.warning("將使用隨機初始化的模型")
        except Exception as e:
            logger.error(f"載入模型失敗: {e}")
            logger.warning("將使用隨機初始化的模型")
    
    def optimize_single_case(self, case_name: str) -> Dict[str, any]:
        """
        優化單個測試案例
        
        Args:
            case_name: 測試案例名稱
        
        Returns:
            優化結果
        """
        logger.info(f"開始優化案例: {case_name}")
        start_time = datetime.now()
        
        try:
            # 重置環境
            environment_state = self.env.reset(case_name)
            initial_tns = environment_state.current_tns
            initial_wns = environment_state.current_wns
            initial_power = environment_state.current_power
            
            # 記錄優化過程
            optimization_steps = []
            actions_taken = []
            successful_actions = 0
            
            # 獲取當前權重用於評分
            tns_weight, power_weight = self.env.get_current_weights()
            
            # 追蹤最佳表現的前三名
            best_performances = []  # 存儲 (step, score, tns_improvement, power_improvement, metrics)
            
            # 記錄初始狀態作為 step 0
            initial_score = 0.0  # 初始狀態評分為 0
            best_performances.append({
                'step': 0,
                'score': initial_score,
                'tns_improvement': 0.0,
                'power_improvement': 0.0,
                'tns': initial_tns,
                'wns': initial_wns,
                'power': initial_power
            })
            
            step_count = 0
            while not environment_state.done and step_count < self.inference_config.max_actions:
                # 獲取二維動作
                action, info = self.agent.get_action(
                    environment_state, 
                    deterministic=self.inference_config.greedy
                )
                
                # 記錄二維動作資訊
                candidate_idx, replacement_idx = action
                candidate_cell = environment_state.candidate_cells[candidate_idx]
                
                # 執行動作
                next_environment_state, reward, done, env_info = self.env.step(action)
                
                # 統計成功動作
                if env_info.get('action_success', False):
                    successful_actions += 1
                
                # 計算當前步驟的改善和評分
                current_tns_improvement = initial_tns - next_environment_state.current_tns
                current_power_improvement = initial_power - next_environment_state.current_power
                
                # 使用權重計算綜合評分（越高越好）
                current_score = (tns_weight * current_tns_improvement + 
                               power_weight * current_power_improvement)
                
                # 記錄步驟詳細資訊
                step_info = {
                    'step': step_count + 1,
                    'action': {
                        'candidate_idx': candidate_idx,
                        'replacement_idx': replacement_idx,
                        'candidate_cell': candidate_cell,
                        'target_cell': env_info.get('target_cell', 'unknown')
                    },
                    'metrics_before': {
                        'tns': environment_state.current_tns,
                        'wns': environment_state.current_wns,
                        'power': environment_state.current_power
                    },
                    'metrics_after': {
                        'tns': next_environment_state.current_tns,
                        'wns': next_environment_state.current_wns,
                        'power': next_environment_state.current_power
                    },
                    'improvements': {
                        'tns': current_tns_improvement,
                        'power': current_power_improvement
                    },
                    'score': current_score,
                    'reward': reward,
                    'success': env_info.get('action_success', False),
                    'log_prob': info.get('log_prob', 0.0),
                    'value': info.get('value', 0.0)
                }
                optimization_steps.append(step_info)
                actions_taken.append(action)
                
                # 更新最佳表現記錄
                performance_record = {
                    'step': step_count + 1,
                    'score': current_score,
                    'tns_improvement': current_tns_improvement,
                    'power_improvement': current_power_improvement,
                    'tns': next_environment_state.current_tns,
                    'wns': next_environment_state.current_wns,
                    'power': next_environment_state.current_power
                }
                
                # 將當前表現插入到最佳表現列表中（保持按評分降序排列）
                best_performances.append(performance_record)
                best_performances.sort(key=lambda x: x['score'], reverse=True)
                best_performances = best_performances[:3]  # 只保留前三名
                
                # 更新狀態
                environment_state = next_environment_state
                step_count += 1
                
                logger.info(f"步驟 {step_count}: 候選{candidate_idx}({candidate_cell})→替換{replacement_idx}, "
                           f"TNS={environment_state.current_tns:.4f}, "
                           f"WNS={environment_state.current_wns:.4f}, "
                           f"Power={environment_state.current_power:.6f}, "
                           f"評分={current_score:.4f}, 獎勵={reward:.4f}, 成功={step_info['success']}")
            
            # 計算優化結果
            final_tns = environment_state.current_tns
            final_wns = environment_state.current_wns
            final_power = environment_state.current_power
            
            tns_improvement = initial_tns - final_tns
            wns_improvement = initial_wns - final_wns
            power_improvement = initial_power - final_power
            success_rate = successful_actions / step_count if step_count > 0 else 0.0
            
            optimization_time = datetime.now() - start_time
            
            # 判斷是否達成目標
            tns_goal_achieved = final_tns >= self.rl_config.tns_goal_threshold
            wns_goal_achieved = final_wns >= self.rl_config.wns_goal_threshold
            convergence = tns_goal_achieved and wns_goal_achieved
            
            # 輸出最佳表現前三名
            logger.info("=" * 60)
            logger.info(f"🏆 最佳表現前三名 (權重 TNS:{tns_weight:.2f}, Power:{power_weight:.2f}):")
            for i, perf in enumerate(best_performances[:3], 1):
                logger.info(f"第{i}名 - 步驟 {perf['step']}: "
                           f"評分={perf['score']:.4f}, "
                           f"TNS改善={perf['tns_improvement']:+.4f}, "
                           f"Power改善={perf['power_improvement']:+.6f}, "
                           f"TNS={perf['tns']:.4f}, Power={perf['power']:.6f}")
            logger.info("=" * 60)
            
            result = {
                'case_name': case_name,
                'initial_metrics': {
                    'tns': initial_tns,
                    'wns': initial_wns,
                    'power': initial_power
                },
                'final_metrics': {
                    'tns': final_tns,
                    'wns': final_wns,
                    'power': final_power
                },
                'improvements': {
                    'tns': tns_improvement,
                    'wns': wns_improvement,
                    'power': power_improvement,
                    'tns_percentage': (tns_improvement / abs(initial_tns)) * 100 if initial_tns != 0 else 0,
                    'wns_percentage': (wns_improvement / abs(initial_wns)) * 100 if initial_wns != 0 else 0,
                    'power_percentage': (power_improvement / initial_power) * 100 if initial_power != 0 else 0
                },
                'optimization_info': {
                    'steps_taken': step_count,
                    'successful_actions': successful_actions,
                    'success_rate': success_rate,
                    'optimization_time': optimization_time.total_seconds(),
                    'convergence': convergence,
                    'tns_goal_achieved': tns_goal_achieved,
                    'wns_goal_achieved': wns_goal_achieved
                },
                'weights_used': {
                    'tns_weight': tns_weight,
                    'power_weight': power_weight
                },
                'best_performances': best_performances,  # 新增：最佳表現前三名
                'optimization_steps': optimization_steps,
                'actions_taken': actions_taken
            }
            
            logger.info(f"優化完成 - TNS: {initial_tns:.4f}→{final_tns:.4f} ({tns_improvement:+.4f}, {result['improvements']['tns_percentage']:+.2f}%), "
                       f"WNS: {initial_wns:.4f}→{final_wns:.4f} ({wns_improvement:+.4f}, {result['improvements']['wns_percentage']:+.2f}%), "
                       f"Power: {power_improvement:+.6f} ({result['improvements']['power_percentage']:+.2f}%), "
                       f"成功率: {success_rate:.2%}, 步數: {step_count}, 時間: {optimization_time.total_seconds():.2f}s")
            
            if convergence:
                logger.info("🎉 時序目標已達成!")
            
            return result
    
        except Exception as e:
            logger.error(f"優化案例 {case_name} 失敗: {e}")
            optimization_time = datetime.now() - start_time
            
            # 返回基本的錯誤結果結構
            return {
                'case_name': case_name,
                'error': str(e),
                'initial_metrics': {
                    'tns': getattr(locals().get('initial_tns'), 'value', 0.0),
                    'wns': getattr(locals().get('initial_wns'), 'value', 0.0),
                    'power': getattr(locals().get('initial_power'), 'value', 0.0)
                },
                'final_metrics': {
                    'tns': 0.0,
                    'wns': 0.0,
                    'power': 0.0
                },
                'improvements': {
                    'tns': 0.0,
                    'wns': 0.0,
                    'power': 0.0,
                    'tns_percentage': 0.0,
                    'wns_percentage': 0.0,
                    'power_percentage': 0.0
                },
                'optimization_info': {
                    'steps_taken': 0,
                    'successful_actions': 0,
                    'success_rate': 0.0,
                    'optimization_time': optimization_time.total_seconds(),
                    'convergence': False,
                    'tns_goal_achieved': False,
                    'wns_goal_achieved': False
                },
                'optimization_steps': [],
                'actions_taken': []
            }
    
    def optimize_multiple_cases(self, case_names: List[str]) -> List[Dict[str, any]]:
        """
        優化多個測試案例
        
        Args:
            case_names: 測試案例名稱列表
        
        Returns:
            優化結果列表
        """
        results = []
        all_best_performances = []  # 收集所有案例的最佳表現
        
        for case_name in case_names:
            try:
                result = self.optimize_single_case(case_name)
                results.append(result)
                
                # 收集最佳表現（添加案例名稱）
                if 'best_performances' in result:
                    for perf in result['best_performances']:
                        perf_with_case = perf.copy()
                        perf_with_case['case_name'] = case_name
                        all_best_performances.append(perf_with_case)
                        
            except Exception as e:
                logger.error(f"優化案例 {case_name} 失敗: {e}")
                # 建立錯誤結果
                error_result = {
                    'case_name': case_name,
                    'error': str(e),
                    'optimization_info': {
                        'steps_taken': 0,
                        'optimization_time': 0,
                        'convergence': False
                    }
                }
                results.append(error_result)
        
        # 總結跨案例的最佳表現前三名
        if all_best_performances:
            # 按評分排序，取前三名
            all_best_performances.sort(key=lambda x: x['score'], reverse=True)
            top_3_overall = all_best_performances[:3]
            
            logger.info("=" * 80)
            logger.info("🌟 跨所有案例的最佳表現前三名:")
            for i, perf in enumerate(top_3_overall, 1):
                logger.info(f"第{i}名 - {perf['case_name']} 步驟 {perf['step']}: "
                           f"評分={perf['score']:.4f}, "
                           f"TNS改善={perf['tns_improvement']:+.4f}, "
                           f"Power改善={perf['power_improvement']:+.6f}")
            logger.info("=" * 80)
            
            # 將跨案例最佳表現添加到結果中
            for result in results:
                if 'error' not in result:
                    result['overall_best_performances'] = top_3_overall
        
        return results
    
    def generate_optimization_actions(self, case_name: str) -> List[OptimizationAction]:
        """
        生成優化動作序列 (不執行，僅生成動作)
        
        Args:
            case_name: 測試案例名稱
        
        Returns:
            優化動作列表
        """
        logger.info(f"生成優化動作序列: {case_name}")
        
        # 重置環境
        state = self.env.reset(case_name)
        actions = []
        
        step_count = 0
        while not state.done and step_count < self.inference_config.max_actions:
            # 獲取狀態向量
            state_vector = self.env.get_state_vector()
            
            # 獲取動作
            action, info = self.agent.get_action(state_vector, deterministic=True)
            
            # 解碼動作為優化動作
            optimization_action = self.env.decode_action(action)
            actions.append(optimization_action)
            
            # 模擬執行 (僅更新狀態)
            next_state, reward, done, env_info = self.env.step(action)
            state = next_state
            step_count += 1
        
        logger.info(f"生成了 {len(actions)} 個優化動作")
        return actions
    
    def save_results(self, results: List[Dict[str, any]], filename: str = None):
        """
        保存優化結果
        
        Args:
            results: 結果列表
            filename: 檔案名稱
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        filepath = os.path.join(self.inference_config.output_dir, filename)
        
        # 計算總結統計
        summary = self._calculate_summary_stats(results)
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'inference_config': {
                'max_actions': self.inference_config.max_actions,
                'greedy': self.inference_config.greedy,
                'temperature': self.inference_config.temperature
            },
            'summary': summary,
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"結果已保存至: {filepath}")
        
        return filepath
    
    def _calculate_summary_stats(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """計算總結統計"""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results'}
        
        tns_improvements = [r['improvements']['tns'] for r in valid_results]
        power_improvements = [r['improvements']['power'] for r in valid_results]
        steps_taken = [r['optimization_info']['steps_taken'] for r in valid_results]
        optimization_times = [r['optimization_info']['optimization_time'] for r in valid_results]
        
        return {
            'total_cases': len(results),
            'successful_cases': len(valid_results),
            'failed_cases': len(results) - len(valid_results),
            'average_tns_improvement': np.mean(tns_improvements),
            'average_power_improvement': np.mean(power_improvements),
            'average_steps': np.mean(steps_taken),
            'average_time': np.mean(optimization_times),
            'best_tns_improvement': np.max(tns_improvements),
            'best_power_improvement': np.max(power_improvements),
            'convergence_rate': sum(1 for r in valid_results if r['optimization_info']['convergence']) / len(valid_results)
        }

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='使用 RL 智能體進行 IC CAD 優化推論')
    parser.add_argument('--model', type=str, required=True, help='模型檔案路徑')
    parser.add_argument('--cases', type=str, nargs='+', help='測試案例名稱')
    parser.add_argument('--max_actions', type=int, default=10, help='最大動作數量')
    parser.add_argument('--greedy', action='store_true', help='使用貪婪策略')
    parser.add_argument('--output_dir', type=str, help='輸出目錄')
    parser.add_argument('--save_actions', action='store_true', help='儲存動作序列而非執行')
    
    args = parser.parse_args()
    
    # 建立配置
    inference_config = default_inference_config
    inference_config.actor_model_path = args.model
    inference_config.max_actions = args.max_actions
    inference_config.greedy = args.greedy
    
    if args.output_dir:
        inference_config.output_dir = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 設定測試案例
    test_cases = args.cases or ['c17', 'c432', 'c499', 'c880']
    
    logger.info(f"開始推論...")
    logger.info(f"模型: {args.model}")
    logger.info(f"測試案例: {test_cases}")
    logger.info(f"最大動作數: {args.max_actions}")
    logger.info(f"貪婪策略: {args.greedy}")
    
    # 建立推論引擎
    engine = InferenceEngine(inference_config)
    
    if args.save_actions:
        # 僅生成動作序列
        all_actions = {}
        for case in test_cases:
            actions = engine.generate_optimization_actions(case)
            all_actions[case] = [
                {
                    'action_type': a.action_type,
                    'target_cell': a.target_cell,
                    'new_cell_type': a.new_cell_type,
                    'position': a.position
                }
                for a in actions
            ]
        
        # 保存動作序列
        actions_file = os.path.join(inference_config.output_dir, 'optimization_actions.json')
        with open(actions_file, 'w') as f:
            json.dump(all_actions, f, indent=2)
        
        logger.info(f"動作序列已保存至: {actions_file}")
    
    else:
        # 執行優化
        results = engine.optimize_multiple_cases(test_cases)
        
        # 保存結果
        results_file = engine.save_results(results)
        
        # 輸出統計
        summary = engine._calculate_summary_stats(results)
        logger.info("=== 優化統計 ===")
        logger.info(f"總案例數: {summary['total_cases']}")
        logger.info(f"成功案例數: {summary['successful_cases']}")
        logger.info(f"平均 TNS 改善: {summary['average_tns_improvement']:.4f}")
        logger.info(f"平均功耗改善: {summary['average_power_improvement']:.6f}")
        logger.info(f"平均步數: {summary['average_steps']:.1f}")
        logger.info(f"收斂率: {summary['convergence_rate']:.2%}")

if __name__ == "__main__":
    main()
