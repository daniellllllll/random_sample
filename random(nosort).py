# -*- coding: utf-8 -*-


import os
import numpy
import matplotlib
from matplotlib import image, pyplot

clear = lambda: os.system('cls')
clear()

FDC = []


means = (20,50,70,50,20)
stds = (10,10,5,5,10)


for i, (mean, std) in enumerate(zip(means, stds)):
  
  x=[]  
  while len(x) < 20:
    sample = numpy.random.normal(loc=mean, scale=std)
    if (sample < mean+std) and (sample > mean-std):
      x.append(sample)
      
  FDC.extend(x)
      

str_FDC = [ str(value) for value in FDC]

matplotlib.rcParams['backend'] = "Qt4Agg"

pyplot.clf()
pyplot.subplot(1,1,1)
pyplot.plot(range(100),FDC, 'yo-')
pyplot.title("test")
pyplot.ylabel("ylabel")
pyplot.show()
    
    
os.chdir("C:\\Users\\Lee\\Desktop\\Python")

f = open("FDC_nosort.csv", "w")
f.write(",".join(str_FDC))
f.close()    

