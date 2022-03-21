#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 08:05:58 2022

@author: caroneddy
"""
import numpy as np

data = np.random.random((int(1e7),))
window_size = int(5e5)
slices = [slice(start, start + window_size)
          for start in range(0, data.size - window_size, int(1e5))]


import time


def slow_mean(data, sl):
    """Simulate a time consuming processing."""
    time.sleep(0.01)
    return data[sl].mean()


tic = time.time()
results0 = np.array([slow_mean(data, sl) for sl in slices])
toc = time.time()
print('\nElapsed time computing the average of couple of slices {:.2f} s'
      .format(toc - tic))


from joblib import Parallel, delayed, dump, load
import os

folder = './joblib_memmap'
try:
    os.mkdir(folder)
except FileExistsError:
    pass

data_filename_memmap = os.path.join(folder, 'data_memmap')
dump(data, data_filename_memmap)
data = load(data_filename_memmap, mmap_mode='r')

tic = time.time()
results2 = Parallel(n_jobs=8)(delayed(slow_mean)(data, sl) for sl in slices)
toc = time.time()
print('\nElapsed time computing the average of couple of slices {:.2f} s\n'
      .format(toc - tic))
