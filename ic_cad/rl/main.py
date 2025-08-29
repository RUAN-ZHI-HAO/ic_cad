"""
IC CAD 2D 動作空間強化學習優化器主程式

這個模組提供了完整的 2D 動作空間強化學習優化流程，
包括訓練、推論和電路優化功能。

作者: AI Assistant
日期: 2024
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

# 加入路徑
sys.path.append('/root/ruan_workspace/ic_cad')

# 匯入自定義模組
from config import RLConfig, InferenceConfig
from rl.train_agent import TwoDimensionalTrainingManager
from rl.inference import TwoDimensionalInferenceEngine
from rl.ppo_agent import TwoDimensionalPPOAgent

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/ruan_workspace/ic_cad/rl/logs/main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TwoDimensionalICCADOptimizer:
    """
    2D 動作空間 IC CAD 強化學習優化器
    
    提供訓練、推論和完整優化流程的統一介面
    """
    
    def __init__(self):
        """初始化優化器"""
        self.rl_config = RLConfig()
        self.inference_config = InferenceConfig()
        
        # 確保必要目錄存在
        self._ensure_directories()
        
        logger.info("TwoDimensionalICCADOptimizer 初始化完成")
        logger.info(f"2D 動作空間配置 - 候選數量: {self.rl_config.max_candidates}, "
                   f"替換數量: {self.rl_config.max_replacements}")
    
    def _ensure_directories(self):
        """確保必要目錄存在"""
        dirs = [
            '/root/ruan_workspace/ic_cad/rl/logs',
            '/root/ruan_workspace/ic_cad/rl/models',
            '/root/ruan_workspace/ic_cad/rl/inference_results',
            '/root/ruan_workspace/ic_cad/rl/full_pipeline_results',
            '/root/ruan_workspace/ic_cad/rl/training_results'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _check_required_files(self) -> bool:
        """檢查必要檔案是否存在"""
        required_files = [
            '/root/ruan_workspace/ic_cad/rl/config.py',
            '/root/ruan_workspace/ic_cad/rl/ppo_agent.py',
            '/root/ruan_workspace/ic_cad/rl/train_agent.py',
            '/root/ruan_workspace/ic_cad/rl/inference.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"缺少必要檔案: {missing_files}")
            return False
        
        return True
    
    def train_rl_agent(self, case_names: List[str], 
                      train_episodes: int = 1000) -> Dict[str, any]:
        """
        訓練 2D 動作空間 RL 代理
        
        Args:
            case_names: 訓練案例名稱列表
            train_episodes: 訓練回合數
            
        Returns:
            訓練結果字典
        """
        logger.info(f"開始訓練 2D 動作空間 RL 代理 - 案例: {case_names}")
        
        # 檢查必要檔案
        if not self._check_required_files():
            raise FileNotFoundError("缺少必要的系統檔案")
        
        # 更新配置
        self.rl_config.max_episodes = train_episodes
        
        # 建立訓練目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_dir = f'/root/ruan_workspace/ic_cad/rl/training_results/{timestamp}'
        os.makedirs(train_dir, exist_ok=True)
        
        # 建立訓練管理器
        trainer = TwoDimensionalTrainingManager(self.rl_config)
        
        # 執行訓練
        training_results = trainer.train(case_names, train_dir)
        
        # 保存最終模型
        model_save_path = os.path.join(train_dir, 'final_model.pth')
        trainer.agent.save_model(model_save_path)
        
        # 準備返回結果
        result = {
            'training_directory': train_dir,
            'model_path': model_save_path,
            'total_episodes': train_episodes,
            'case_names': case_names,
            'training_results': training_results,
            'config': {
                'max_candidates': self.rl_config.max_candidates,
                'max_replacements': self.rl_config.max_replacements,
                'learning_rate': self.rl_config.learning_rate,
                'batch_size': self.rl_config.batch_size
            }
        }
        
        logger.info(f"2D 動作空間 RL 代理訓練完成 - 模型保存至: {model_save_path}")
        
        return result
    
    def optimize_circuit(self, case_names: List[str], 
                        model_path: Optional[str] = None,
                        max_actions: int = 10,
                        save_detailed: bool = True) -> Dict[str, any]:
        """
        使用訓練好的 2D 動作空間模型優化電路
        
        Args:
            case_names: 電路案例名稱列表
            model_path: 模型路徑
            max_actions: 最大優化動作數
            save_detailed: 是否保存詳細結果
            
        Returns:
            優化結果
        """
        logger.info(f"開始 2D 動作空間電路優化: {case_names}")
        
        # 設定模型路徑
        if model_path is None:
            model_path = '/root/ruan_workspace/ic_cad/rl/models/best_model.pth'
        
        if not os.path.exists(model_path):
            logger.warning(f"RL 模型不存在: {model_path}")
            logger.warning("將使用隨機初始化的模型進行優化")
        
        # 更新推論配置
        self.inference_config.max_actions = max_actions
        
        # 建立推論引擎
        engine = TwoDimensionalInferenceEngine(self.inference_config, self.rl_config)
        
        # 載入模型
        if os.path.exists(model_path):
            engine.load_model(model_path)
        
        # 執行優化
        results = engine.optimize_multiple_cases(case_names)
        
        # 保存結果
        if save_detailed:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f'/root/ruan_workspace/ic_cad/rl/inference_results/optimization_results_{timestamp}.json'
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"詳細結果已保存至: {results_file}")
        
        # 計算總結統計
        summary = self._calculate_summary_stats(results)
        
        return {
            'summary': summary,
            'detailed_results': results if save_detailed else None,
            'model_used': model_path,
            'case_names': case_names,
            'optimization_config': {
                'max_actions': max_actions,
                'max_candidates': self.rl_config.max_candidates,
                'max_replacements': self.rl_config.max_replacements
            }
        }
    
    def _calculate_summary_stats(self, results: List[Dict]) -> Dict[str, any]:
        """計算總結統計"""
        if not results:
            return {}
        
        tns_improvements = [r['improvements']['tns'] for r in results if 'improvements' in r]
        wns_improvements = [r['improvements']['wns'] for r in results if 'improvements' in r]
        power_improvements = [r['improvements']['power'] for r in results if 'improvements' in r]
        success_rates = [r['optimization_info']['success_rate'] for r in results if 'optimization_info' in r]
        convergence_rates = [r['optimization_info']['convergence'] for r in results if 'optimization_info' in r]
        
        return {
            'total_cases': len(results),
            'successful_optimizations': len(tns_improvements),
            'average_tns_improvement': np.mean(tns_improvements) if tns_improvements else 0,
            'average_wns_improvement': np.mean(wns_improvements) if wns_improvements else 0,
            'average_power_improvement': np.mean(power_improvements) if power_improvements else 0,
            'average_success_rate': np.mean(success_rates) if success_rates else 0,
            'convergence_rate': np.mean(convergence_rates) if convergence_rates else 0,
            'cases_converged': sum(convergence_rates) if convergence_rates else 0,
            'best_tns_improvement': max(tns_improvements) if tns_improvements else 0,
            'best_wns_improvement': max(wns_improvements) if wns_improvements else 0
        }
    
    def full_pipeline(self, case_names: List[str], train_episodes: int = 1000, 
                     max_actions: int = 10, output_dir: str = None) -> Dict[str, any]:
        """
        執行完整流程：訓練 -> 推論 -> 優化
        
        Args:
            case_names: 電路案例名稱列表
            train_episodes: 訓練回合數
            max_actions: 最大優化動作數
            output_dir: 輸出目錄（可選）
            
        Returns:
            完整流程結果
        """
        logger.info("開始執行完整 2D 動作空間 RL 優化流程")
        
        # 準備輸出目錄
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'/root/ruan_workspace/ic_cad/rl/full_pipeline_results/{timestamp}'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 步驟 1: 訓練 RL 代理
        logger.info("步驟 1: 訓練 2D 動作空間 RL 代理")
        train_result = self.train_rl_agent(case_names, train_episodes)
        
        # 保存訓練結果
        train_result_file = os.path.join(output_dir, 'train_results.json')
        with open(train_result_file, 'w') as f:
            json.dump(train_result, f, indent=2)
        
        # 步驟 2: 優化電路
        logger.info("步驟 2: 使用訓練好的 2D 模型優化電路")
        best_model_path = train_result['model_path']
        optimization_result = self.optimize_circuit(
            case_names, 
            model_path=best_model_path,
            max_actions=max_actions,
            save_detailed=True
        )
        
        # 保存優化結果
        optimization_result_file = os.path.join(output_dir, 'optimization_results.json')
        with open(optimization_result_file, 'w') as f:
            json.dump(optimization_result, f, indent=2)
        
        # 步驟 3: 生成總結報告
        logger.info("步驟 3: 生成總結報告")
        summary_report = self._generate_pipeline_report(
            train_result, optimization_result, output_dir
        )
        
        # 保存總結報告
        report_file = os.path.join(output_dir, 'pipeline_summary.json')
        with open(report_file, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        logger.info(f"完整 2D 動作空間流程完成，結果保存在: {output_dir}")
        
        return {
            'output_directory': output_dir,
            'train_result': train_result,
            'optimization_result': optimization_result,
            'summary_report': summary_report
        }
    
    def _generate_pipeline_report(self, train_result: Dict, optimization_result: Dict, 
                                 output_dir: str) -> Dict[str, any]:
        """生成流程總結報告"""
        report = {
            'pipeline_info': {
                'completion_time': datetime.now().isoformat(),
                'output_directory': output_dir,
                '2d_action_space': True,
                'max_candidates': self.rl_config.max_candidates,
                'max_replacements': self.rl_config.max_replacements
            },
            'training_summary': {
                'total_episodes': train_result.get('total_episodes', 'N/A'),
                'final_reward': train_result.get('final_reward', 'N/A'),
                'convergence': train_result.get('convergence', 'N/A'),
                'model_path': train_result.get('model_path', 'N/A')
            },
            'optimization_summary': optimization_result.get('summary', {}),
            'performance_metrics': {
                'total_cases_optimized': len(optimization_result.get('case_names', [])),
                'overall_success': optimization_result.get('summary', {}).get('convergence_rate', 0) > 0.5,
                '2d_action_performance': {
                    'average_candidate_utilization': 'TBD',
                    'average_replacement_utilization': 'TBD'
                }
            }
        }
        
        return report


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description='IC CAD 2D 動作空間強化學習優化器')
    parser.add_argument('--mode', choices=['train', 'optimize', 'full'], 
                       default='full', help='執行模式')
    parser.add_argument('--cases', nargs='+', required=True,
                       help='電路案例名稱列表')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='訓練回合數')
    parser.add_argument('--max-actions', type=int, default=10,
                       help='最大優化動作數')
    parser.add_argument('--model-path', type=str,
                       help='模型路徑（僅限 optimize 模式）')
    parser.add_argument('--output-dir', type=str,
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    # 建立優化器
    optimizer = TwoDimensionalICCADOptimizer()
    
    try:
        if args.mode == 'train':
            result = optimizer.train_rl_agent(args.cases, args.episodes)
            print(f"2D 動作空間訓練完成，模型保存至: {result['model_path']}")
            
        elif args.mode == 'optimize':
            result = optimizer.optimize_circuit(
                args.cases, 
                model_path=args.model_path,
                max_actions=args.max_actions,
                save_detailed=True
            )
            print(f"2D 動作空間優化完成，總結: {result['summary']}")
            
        elif args.mode == 'full':
            result = optimizer.full_pipeline(
                args.cases,
                train_episodes=args.episodes,
                max_actions=args.max_actions,
                output_dir=args.output_dir
            )
            print(f"完整 2D 動作空間流程完成，結果目錄: {result['output_directory']}")
            
    except Exception as e:
        logger.error(f"執行錯誤: {e}")
        raise


if __name__ == "__main__":
    main()
