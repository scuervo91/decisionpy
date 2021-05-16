from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import List, Union, Optional, Callable, Tuple
from enum import Enum
import numpy as np
#Localimports
from ..utility import UtilityFunction

class NodeType(str, Enum):
    decision_node = 'decision'
    random_node = 'random'
    end = 'end'
    

class Node(BaseModel):
    name: str = Field(...)
    type: NodeType = Field(...)
    value: float = Field(0)
    expected_value: Optional[float] = Field(None)
    cum_value: float = Field(0)
    children: Optional[List[Node]] = Field(None)
    utility_function: UtilityFunction = Field(UtilityFunction())
    expected_value_func: Optional[Callable[...,Tuple]] = Field(None)
    probability: float = Field(1, ge=0, le=1)
    path: Optional[str] = Field(None)
    
    @validator('expected_value_func',always=True)
    def check_expected_value_func(cls,v,values):
        if v is None:
            if values['type'] == 'decision':
                return lambda x,y: (np.max(x),np.argmax(x))
            if values['type'] == 'random':
                return lambda x,y: (np.dot(x,y),)
            
    class Config:
        arbitrary_types_allowed = True
        extra = 'forbid'
        validate_assignment = True
        
    def solve(self):
        #If the expected value is already estimated
        if self.expected_value:
            return self.expected_value
        
        if self.children is None:
            self.expected_value = self.utility_function.get_utility(self.value + self.cum_value)
            return self.expected_value
               
        children_values = []
        children_prob = []
        for c in self.children:
            print(c.name)
            c.cum_value = self.cum_value
            child_value = c.solve()
            children_values.append(child_value)
            children_prob.append(c.probability)
        c_values = np.array(children_values, dtype='float')
        p_values = np.array(children_prob, dtype='float')
        
        cu_values = self.utility_function.get_utility(c_values)
        result = self.expected_value_func(cu_values,p_values)
        
        if self.type == 'decision':
            self.path = self.children[result[1]].name
            
        return result[0]

Node.update_forward_refs()
