# cell_replacement_manager.py
"""
Cell Replacement Manager
管理 cell 的替換選項和索引映射
"""
import json
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CellReplacementManager:
    """
    管理 cell 替換的類別
    
    職責：
    1. 載入 cell groups JSON
    2. 建立 cell_name -> group_index 的映射
    3. 提供查詢介面：給定 cell_name，返回可替換的選項列表
    4. 提供 RL 動作解碼：(group_index, cell_index_in_group) -> cell_name
    """
    
    def __init__(self, json_file_path: str):
        """
        初始化 CellReplacementManager
        
        Args:
            json_file_path: JSON 文件路徑，包含 cell groups
        """
        self.json_file_path = json_file_path
        self.cell_groups = []
        self.cell_to_group = {}  # cell_name -> group_index 的映射
        
        # 定義不可替換的 cell 類型
        self.non_replaceable_patterns = [
            'DECAP',      # 去耦合電容器
            'FILLER',     # 填充 cells
            'TAPCELL',    # 基板接觸 cells
            'sram_',      # SRAM cells
        ]
        
        self._load_groups()
        self._build_index()
        
    def _load_groups(self):
        """載入 cell groups JSON"""
        try:
            with open(self.json_file_path, 'r') as f:
                self.cell_groups = json.load(f)
            
            # 計算最大 group 大小（用於 action mask）
            self.max_group_size = max(len(group) for group in self.cell_groups) if self.cell_groups else 0
            
            # 🎯 自動檢測第 0 個群組大小作為推薦的 max_replacements
            self.recommended_max_replacements = len(self.cell_groups[0]) if self.cell_groups else 84
            
            logger.info(f"Loaded {len(self.cell_groups)} cell groups")
            logger.info(f"Max group size: {self.max_group_size}")
            logger.info(f"Recommended max_replacements (from group 0): {self.recommended_max_replacements}")
            
        except Exception as e:
            logger.error(f"Failed to load cell groups from {self.json_file_path}: {e}")
            raise
    
    def get_recommended_max_replacements(self) -> int:
        """
        獲取推薦的 max_replacements 值
        基於第 0 個群組大小（通常是最大的群組）
        
        Returns:
            推薦的 max_replacements 值
        """
        return getattr(self, 'recommended_max_replacements', 84)
    
    def _build_index(self):
        """建立 cell_name -> group_index 的映射"""
        self.cell_to_group = {}
        
        for group_idx, group in enumerate(self.cell_groups):
            for cell_name in group:
                if cell_name in self.cell_to_group:
                    logger.warning(f"Duplicate cell name: {cell_name}")
                self.cell_to_group[cell_name] = group_idx
        
        logger.info(f"Built index for {len(self.cell_to_group)} cells")
    
    def get_replacement_options(self, cell_name: str) -> List[str]:
        """
        獲取指定 cell 的所有替換選項
        
        Args:
            cell_name: 要查詢的 cell 名稱
            
        Returns:
            可替換的 cell 名稱列表（不包括自己）
        """
        if cell_name not in self.cell_to_group:
            logger.warning(f"Cell {cell_name} not found in groups")
            return []
        
        group_idx = self.cell_to_group[cell_name]
        group = self.cell_groups[group_idx]
        
        # 返回同組的其他 cell（排除自己）
        return [c for c in group if c != cell_name]
    
    def get_group_index(self, cell_name: str) -> Optional[int]:
        """獲取 cell 所屬的 group 索引"""
        return self.cell_to_group.get(cell_name)
    
    def get_cell_index_in_group(self, cell_name: str) -> Optional[int]:
        """獲取 cell 在其 group 中的索引"""
        group_idx = self.get_group_index(cell_name)
        if group_idx is None:
            return None
        
        group = self.cell_groups[group_idx]
        try:
            return group.index(cell_name)
        except ValueError:
            return None
    
    def decode_action(self, group_idx: int, cell_idx_in_group: int) -> Optional[str]:
        """
        從動作索引解碼出 cell 名稱
        
        Args:
            group_idx: group 索引
            cell_idx_in_group: 在 group 中的 cell 索引
            
        Returns:
            對應的 cell 名稱，如果無效則返回 None
        """
        if group_idx >= len(self.cell_groups):
            return None
        
        group = self.cell_groups[group_idx]
        if cell_idx_in_group >= len(group):
            return None
        
        return group[cell_idx_in_group]
    
    def get_action_mask_for_candidates(self, candidate_cells: List[str]) -> Tuple[List[int], List[List[bool]]]:
        """
        為候選 cell 列表生成動作 mask
        
        Args:
            candidate_cells: 候選 cell 名稱列表
            
        Returns:
            Tuple[
                List[int]: 候選 cell 對應的 group 索引列表
                List[List[bool]]: 每個候選 cell 的替換選項 mask [n_candidates, max_group_size]
            ]
        """
        candidate_group_indices = []
        action_masks = []
        
        for cell_name in candidate_cells:
            group_idx = self.get_group_index(cell_name)
            
            if group_idx is None:
                # 如果找不到 group，設置為無效
                candidate_group_indices.append(-1)
                action_masks.append([False] * self.max_group_size)
                continue
            
            candidate_group_indices.append(group_idx)
            
            # 建立該 cell 的替換選項 mask
            group = self.cell_groups[group_idx]
            mask = [False] * self.max_group_size
            
            # 標記該 group 中的有效替換選項（排除自己）
            for i, replacement_cell in enumerate(group):
                if replacement_cell != cell_name:  # 不能替換成自己
                    mask[i] = True
            
            action_masks.append(mask)
        
        return candidate_group_indices, action_masks
    
    def get_stats(self) -> Dict[str, int]:
        """獲取統計資訊"""
        group_sizes = [len(group) for group in self.cell_groups]
        return {
            "total_groups": len(self.cell_groups),
            "total_cells": len(self.cell_to_group_idx),
            "max_group_size": self.max_group_size,
            "avg_group_size": sum(group_sizes) / len(group_sizes) if group_sizes else 0,
            "min_group_size": min(group_sizes) if group_sizes else 0
        }


# 測試函數
if __name__ == "__main__":
    manager = CellReplacementManager("/root/cell_groups.json")
    
    # 測試查詢
    test_cell = "NAND2x1_ASAP7_75t_L"
    options = manager.get_replacement_options(test_cell)
    print(f"替換選項 for {test_cell}: {options}")
    
    # 測試動作解碼
    group_idx = manager.get_group_index(test_cell)
    cell_idx = manager.get_cell_index_in_group(test_cell)
    print(f"{test_cell} -> group_idx: {group_idx}, cell_idx: {cell_idx}")
    
    # 測試候選 mask
    candidates = ["NAND2x1_ASAP7_75t_L", "INVx1_ASAP7_75t_L", "BUFx2_ASAP7_75t_L"]
    group_indices, masks = manager.get_action_mask_for_candidates(candidates)
    print(f"候選 cells: {candidates}")
    print(f"Group indices: {group_indices}")
    print(f"Action masks shape: {len(masks)} x {len(masks[0]) if masks else 0}")
    
    # 統計資訊
    stats = manager.get_stats()
    print(f"統計: {stats}")
