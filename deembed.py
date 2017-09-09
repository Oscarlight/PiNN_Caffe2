# read in S parameter, bias and frequency infromation,
# deembed parasitic resistors and inductors, then output deembeded Y parameters
# Reference: 
#  Deembedding method:
#  	http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=3650
#  How deembedding fit into the ANN device modeling:
#    http://www.keysight.com/upload/cmc_upload/All/NeuroFET_Webcast_Final.pdf?&cc=US&lc=eng

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 19:16:48 2017

@author: jashansinghal
"""

from numpy import *
import scipy.linalg

f = 10
s11 = 1
s12 = 2
s21 = 3
s22 = 4

omega=2*3.14*f
Lg=1
Ld=1
Rg=1
Rd=1


S = array([[s11,s12],[s21,s22]])

def deembed(Lg,Ld,Rg,Rd,S):
    I = array([[1,0],[0,1]])
    Z = (I+S).dot(linalg.inv(I-S))
    Znew = array([[Z[0,0]-1j*omega*Lg-Rg,Z[0,1]],[Z[1,0],Z[1,1]-1j*omega*Ld-Rd]])
    Yfinal = linalg.inv(Znew)
    return Yfinal
