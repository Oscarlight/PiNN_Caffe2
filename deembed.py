#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 19:16:48 2017

@author: jashansinghal
"""

from numpy import *
import scipy.linalg
from parser import ac_s_input


def deembed(file_name,lg,ld,rg,rd):
    s11arr,s12arr,s21arr,s22arr,freq,vg,vd= ac_s_input(file_name)
    yfinal = []

    for i in range (0, size(s11arr)):
        s11 = s11arr[i]
        s12 = s12arr[i]
        s21 = s21arr[i]
        s22 = s22arr[i]
        omega = 2*3.14*freq[i]
        s = array([[s11,s12],[s21,s22]])
        imat = array([[1,0],[0,1]])
        z = (imat+s).dot(linalg.inv(imat-s))
        znew = array([
            [z[0,0]-1j*omega*lg-rg, z[0,1]],
            [z[1,0],z[1,1]-1j*omega*ld-rd]])

        yfinal.append(linalg.inv(znew))


    return yfinal

if __name__ == '__main__':
    lg=1
    ld=1
    rg=1
    rd=1
    
    print(deembed('./HEMT_bo/s_vs_f_at_VgVd.mdm',lg,ld,rg,rd))
    





