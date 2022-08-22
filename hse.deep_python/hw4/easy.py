from hw_1.fibonacci import fib
from time import perf_counter
from threading import Thread
from multiprocessing import Process

n = 10**5

s = perf_counter()
for _ in range(10):
    fib(n)
simple_time = perf_counter() - s

s = perf_counter()
threads = []
for _ in range(10):
    threads.append(Thread(target=fib, args=(n,)))
    threads[-1].start()

for t in threads:
    t.join()
threads_time = perf_counter() - s

s = perf_counter()
processes = []
for _ in range(10):
    processes.append(Process(target=fib, args=(n,)))
    processes[-1].start()

for p in processes:
    p.join()

processes_time = perf_counter() - s

with open('artifacts/easy.txt', 'w') as f:
    f.write(f"Sync time: {simple_time}, s\n"
            f"Thread time: {threads_time}, s\n"
            f"Process time: {processes_time}, s")
