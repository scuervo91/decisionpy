from pydantic import BaseModel, Field
from typing import Callable, List, Union
import types
import numpy as np
from scipy import optimize


class UtilityFunction(BaseModel):
    name: str = Field(None)
    func: Callable[[int], str] = Field(lambda x: np.atleast_1d(x))
    
    class Config:
        json_encoders = {
            types.FunctionType: str,
        }
    
    def get_utility(self,x):
        return self.func(x)
    

class Alternative(BaseModel):
    name: str = Field(None)
    probability: float = Field(...,ge=0,le=1)
    value: "Union[float,Lotery]" = Field(...)
    
    def get_value(self):
        if isinstance(self.value,Lotery):
            return self.value.ev()
        return self.value
    

class Lotery(BaseModel):
    alternatives: List[Alternative] = Field(...)
    
    
    def max_value(self):
        v = []
        for a in self.alternatives:
            v.append(a.value)
        return np.max(v)

    def min_value(self):
        v = []
        for a in self.alternatives:
            v.append(a.value)
        return np.min(v)
    #Expected Value
    def exp_value(self,b=0):
        p = []
        v = []
        for a in self.alternatives:
            p.append(a.probability)
            v.append(a.value)
        return np.dot(p,np.atleast_1d(v)+b)
    
    #Expected Value Utility Function
    def exp_value_u(self,uf:UtilityFunction,b=0):
        p = []
        v = []
        for a in self.alternatives:
            p.append(a.probability)
            v.append(a.value)
        return np.dot(p,uf.get_utility(np.atleast_1d(v)-b))        
    
    # Certainty Equivalent
    def emc(self,uf:UtilityFunction):
        
        def cost_func(x):
            return uf.get_utility(x) - self.exp_value_u(uf)
        
        return optimize.root_scalar(cost_func, bracket=[self.min_value(), self.max_value()],method='brentq')
    
    # Risk Premium
    def risk_premium(self,uf:UtilityFunction):
        return self.exp_value() - self.emc(uf).root
    
    #purchase price
    def purchase_price(self, uf:UtilityFunction):
        def cost_func(x):
            return self.exp_value_u(uf,b=x) - uf.get_utility(0)
        
        return optimize.root_scalar(cost_func, bracket=[self.min_value(), self.max_value()],method='brentq')
 
        
        
Alternative.update_forward_refs()
Lotery.update_forward_refs()