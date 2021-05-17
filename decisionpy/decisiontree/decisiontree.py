from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import List, Union, Optional, Callable, Tuple
from enum import Enum
import numpy as np
from pydantic.typing import new_type_supertype
from rich.console import Console 
from rich.tree import Tree
from rich.theme import Theme 
from rich.text import Text
from rich.layout import Layout
from rich.panel import Panel

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
    cum_value: Optional[float] = Field(None)
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
        
    def solve(self, init=True):
        
        if init:
            self.cum_value = self.value 
            init = False
        
        #If the expected value is already estimated
        if self.expected_value:
            return self.expected_value
        
        if self.children is None or self.type =='end':
            self.expected_value = self.utility_function.get_utility(self.cum_value)
            return self.expected_value
               
        children_values = []
        children_prob = []
        for c in self.children:
            c.cum_value = self.cum_value + c.value
            child_value = c.solve(init=False)
            children_values.append(child_value)
            children_prob.append(c.probability)
        c_values = np.array(children_values, dtype='float')
        p_values = np.array(children_prob, dtype='float')
        
        cu_values = self.utility_function.get_utility(c_values)
        result = self.expected_value_func(cu_values,p_values)
        
        if self.type == 'decision':
            self.path = self.children[result[1]].name
            
        self.expected_value = result[0]
        return self.expected_value
    
    def tree(self, decision=None, expand_row=False):
        if self.type == 'decision':
            root_text = f":black_large_square: [bold][u]{self.name}[/u][/bold]"
        elif self.type == 'random':
            root_text = f":white_circle:[bold][u]{self.name}[/u][/bold]"
        else:
            root_text = f":small_red_triangle_down: [bold][u]{self.name}[/u][/bold]"
        
        new_line = '\n' if expand_row else ''
        
        prob_text = f"{new_line} :game_die: [bold cyan]Prob[/bold cyan]: [u]{self.probability}[/u] |" if self.probability < 1 else ""
        value_text = f"{new_line} :100: [bold blue]Value[/bold blue]: [u]{self.value}[/u] |"
        cumvalue_text = f"{new_line} :heavy_plus_sign: [bold magenta]Cum Value[/bold magenta]: [u]{self.cum_value}[/u] |" if self.cum_value is not None else ""
        expected_value_text = f"\n     :star: [bold green]Exp. Value[/bold green]: [u]{self.expected_value}[/u] |" if self.expected_value is not None else ""
        
        if decision is not None:
            if decision ==  self.name:
                dec_emoji = ":white_check_mark:"
            else:
                dec_emoji = ":no_entry_sign:"
        else:
            dec_emoji = ""
        
        tree_text = dec_emoji + root_text + prob_text + value_text + cumvalue_text + expected_value_text
        node_tree = Tree(
            tree_text,
            highlight=True,
            guide_style=f"{'underline2' if self.type=='random' else 'bold'}"
        )

        if self.children is None or self.type =='end':
            return node_tree 
        
        for c in self.children:
            node_tree.add(c.tree(decision=self.path, expand_row=expand_row))
                   
        return node_tree


Node.update_forward_refs()
