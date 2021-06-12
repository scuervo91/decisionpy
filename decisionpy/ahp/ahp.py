from turtle import shape
import numpy as np 
from pydantic import BaseModel,  validator
from typing import List

saaty_cr_table = {
    1:0,
    2:0,
    3:0.58,
    4:0.9,
    5:1.12,
    6:1.24,
    7:1.32,
    8:1.41,
    9:1.45,
    10:1.49,
    11:1.51,
    12:1.48,
    13:1.56,
    14:1.57,
    15:1.59
}

class Matrix(BaseModel):
    values: List[List[float]]
    
    class Config:
        extra = 'forbid'
        validate_assignment = True

    @validator('values',always=True)
    def check_dims(cls,v):
        A = np.atleast_2d(v)
        A_shape = A.shape
        assert A_shape[0] == A_shape[1]
        return v

    def get_matrix(self):
        return np.atleast_2d(self.values)
    
    def shape(self):
        return self.get_matrix().shape[0]
    
    def get_eigen(self):
        #get the matrix
        A = self.get_matrix()
        
        # get the eigen values and eigen vector
        w,v = np.linalg.eig(A)
        
        # get the maximum eigenvalue index
        ind_max = np.argmax(w)

        #extract the maximum eigenvalue and its corresponding eigenvector
        eigva = w[ind_max]
        eigvec = v[:,ind_max]
        
        return eigva, eigvec.real
    
    def get_weights(self):
        _, eigvec = self.get_eigen()
        
        #Normalize the eigenvector in order to sum 1
        eigvec_norm = eigvec / eigvec.sum()
        
        return eigvec_norm
    
    def consistency(self):
        eigva, _ = self.get_eigen()
        n = self.shape()
        return (eigva - n)/(n - 1)
    
    def consistency_ratio(self):
        ci = self.consistency()
        cn =  saaty_cr_table[self.shape()]

        return ci/cn
        
    
    