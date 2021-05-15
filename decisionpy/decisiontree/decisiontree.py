from __future__ import annotations
from unittest.mock import Base
from pydantic import BaseModel
from typing import List, Union, Optional
from enum import Enum


class NodeType(str, Enum):
    decision_node = 'decision_node'
    random_node = 'random_node'
    

class Node(BaseModel):
    name: str
    value: Optional[float] =None
    type: NodeType
    children: Optional[List[Node]] =None
    
    
    class Config:
        arbitrary_types_allowed = True
        extra = 'forbid'
                    
            
class Tree(BaseModel):
    name: str
    children: Optional[List[Node]] =None