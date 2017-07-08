# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:28:16 2015

@author: daniel
"""
import os
from scipy import stats
import numpy as np
import pylab

#clear = lambda: os.system('clear')
#clear()

#samples = [2,3,4,5,6,7,8,10]
#samples= open("FDC.csv", "r"). read().strip().split(",")
#samples = [ float(value) for value in samples]


def linear(y, x = None):
        if x is None:
            x = range(1, len(y)+1)
        x = list(x)
        
        slope, intercept, r_value, _,slope_std_error = stats.linregress(x,y)
        slope=np.array(slope)

        
        predict_y = intercept + slope*x
        pred_error = y - predict_y
        degress_of_freedom = len(x)-2
        residual_std_error = np.sqrt(np.sum(pred_error**2)/degress_of_freedom)
        
        return slope, intercept, residual_std_error
  
if __name__ == "__main__":
    samples = [2,3,4,5,6,7,8,9]
    slope, intercept, std = linear(samples)
    assert std == 0, "std error"
    assert slope == 1, "slope error"
    assert intercept == 1, "intercept error"
    
    samples = [1,2,6]
    slope, intercept, std = linear(samples)
    assert abs(std - 1.22474487139) < 0.00001, "std error"
    assert abs(slope - 2.5) < 0.00001, "slope error"
    assert abs(intercept - -2) < 0.0001, "intercept error"
    print("data correct")
