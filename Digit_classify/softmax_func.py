#!/usr/bin/env python
from scipy import *

def softmax(v):
    expsum=0
    for vi in v:
    	expsum+=exp(vi)
    return array([exp(vi)/expsum for vi in v])

if __name__=='__main__':
	v=[1, 2, 3, 4, 1, 2, 3]
	pv=softmax(v)
	print pv
