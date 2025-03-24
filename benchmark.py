# benchmark.py
import numpy as np
import threading
import time
import os
import multiprocessing

# 导入我们的Cython模块（根据您成功编译的模块名调整）
try:
    from cpu_bound import compute_with_gil, compute_nogil
except ImportError:
    from cpu_bound_simple import compute_with_gil, compute_nogil

ARRAY_SIZE = 1_000_000  # 增大数组大小
ITERATIONS = 10_000     # 增加迭代次数

def test_with_gil(num_threads):
    """使用普通Python线程（受GIL限制）"""
    data = np.ones(ARRAY_SIZE, dtype=np.float64)
    threads = []
    
    def worker():
        compute_with_gil(data, ITERATIONS)
    
    start_time = time.time()
    
    # 创建并启动指定数量的线程
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    return total_time

def test_nogil_threading(num_threads):
    """使用Cython nogil线程（在Python层创建多线程）"""
    data = np.ones(ARRAY_SIZE, dtype=np.float64)
    threads = []
    
    def worker():
        # 注意：每个线程内部没有并行化
        compute_nogil(data, ITERATIONS, num_threads=1)
    
    start_time = time.time()
    
    # 创建并启动指定数量的线程
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    return total_time

def test_nogil_internal(num_threads):
    """使用Cython内部的并行化（OpenMP）"""
    data = np.ones(ARRAY_SIZE, dtype=np.float64)
    
    start_time = time.time()
    # 在Cython函数内部使用指定数量的线程
    compute_nogil(data, ITERATIONS, num_threads=num_threads)
    total_time = time.time() - start_time
    
    return total_time

def run_benchmarks():
    print(f"{'='*60}")
    print(f"性能基准测试 (数组大小: {ARRAY_SIZE}, 迭代次数: {ITERATIONS})")
    print(f"{'='*60}")
    print(f"CPU核心数: {multiprocessing.cpu_count()}")
    print(f"{'='*60}")
    
    # 根据CPU核心数确定要测试的线程数
    thread_counts = [1, 2, 4]
    if multiprocessing.cpu_count() >= 8:
        thread_counts.append(8)
        thread_counts.append(16)
    
    # 单线程基准（作为参考）
    print("运行单线程基准测试...")
    single_time = test_with_gil(1)
    print(f"单线程基准时间: {single_time:.4f}秒")
    print(f"{'-'*60}")
    
    # 测试和打印结果
    print(f"{'线程数':^10}|{'GIL线程':^15}|{'nogil线程':^15}|{'nogil内部':^15}")
    print(f"{'-'*10}|{'-'*15}|{'-'*15}|{'-'*15}")
    
    for num_threads in thread_counts:
        print(f"测试 {num_threads} 线程...")
        
        # 测试带GIL的普通线程
        gil_time = test_with_gil(num_threads)
        # 测试在Python层创建的nogil线程
        nogil_thread_time = test_nogil_threading(num_threads)
        # 测试在Cython内部使用OpenMP的并行
        nogil_internal_time = test_nogil_internal(num_threads)
        
        # 计算加速比
        gil_speedup = single_time / gil_time
        nogil_thread_speedup = single_time / nogil_thread_time
        nogil_internal_speedup = single_time / nogil_internal_time
        
        print(f"{num_threads:^10}|"
              f"{gil_time:.4f}s ({gil_speedup:.2f}x)|"
              f"{nogil_thread_time:.4f}s ({nogil_thread_speedup:.2f}x)|"
              f"{nogil_internal_time:.4f}s ({nogil_internal_speedup:.2f}x)")
    
    print(f"{'='*60}")
    print("注释:")
    print("1. GIL线程: 普通Python线程，受GIL限制")
    print("2. nogil线程: 每个Python线程中都使用nogil但内部单线程")
    print("3. nogil内部: 单个Python线程，但在Cython内部使用OpenMP并行")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_benchmarks()
