# cpu_bound_simple.pyx
# cython: language_level=3
import numpy as np
cimport numpy as np
from cython.parallel import prange
import cython

# 不释放GIL的版本
def compute_with_gil(double[:] data, int iterations):
    cdef:
        Py_ssize_t i
        int n = data.shape[0]
        double result = 0.0
    
    for i in range(n):
        # 简单累加元素的平方，避免复杂表达式
        result += data[i] * data[i] * iterations
    
    return result

# 释放GIL的版本
@cython.boundscheck(False)
@cython.wraparound(False)
def compute_nogil(double[:] data, int iterations, int num_threads=1):
    cdef:
        Py_ssize_t i
        int n = data.shape[0]
        double result = 0.0
    
    # 使用nogil释放GIL
    with nogil:
        # 使用prange进行并行循环
        for i in prange(n, num_threads=num_threads):
            # 简单计算以避免复杂表达式
            result += data[i] * data[i] * iterations
    
    return result
