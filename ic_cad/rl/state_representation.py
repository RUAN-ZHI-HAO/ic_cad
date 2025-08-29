# state_representation.py
from dataclasses import dataclass
from typing import List, Any, Optional

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
    candidate_cells: List[CandidateCell]
    current_tns: float
    current_wns: float
    current_power: float
    step_count: int
    circuit_metrics: Any
    candidate_gnn_features: List[List[float]]
    initial_tns: Optional[float] = None
    initial_wns: Optional[float] = None
    initial_power: Optional[float] = None
