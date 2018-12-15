#!/usr/bin/env python

import numpy

def generate_mask(signal_length,scales,coi_factor):
    result=numpy.zeros([len(scales),signal_length])
    for i in range(0,result.shape[0]):
        for j in range(0,result.shape[1]):
            if j<coi_factor*scales[i] or signal_length-j<coi_factor*scales[i]:
                result[i,j]=0
            else:
                result[i,j]=1
    return result
