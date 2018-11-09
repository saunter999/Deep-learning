#!/usr/bin/env python
import numpy as np
a=[[1,2]]
res=np.zeros((2,3))
for seq in a:
    print seq,type(seq)
    res[1,seq]=1.;print res
#print res
