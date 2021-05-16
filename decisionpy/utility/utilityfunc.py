from pydantic import BaseModel, Field
from typing import Callable


class UtilityFunction(BaseModel):
    func: Callable[[int], str] = Field(lambda x: x)
    
    def get_utility(self,x):
        return self.func(x)
    
    