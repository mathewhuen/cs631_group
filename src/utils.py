import time
from contextlib import ContextDecorator


class AverageDict(dict):
    def calculate(self):
        output = dict()
        for key in self.keys():
            tot = sum(self[key])
            output[key] = {"total": tot, "mean": tot / len(self[key])}
        return output


class TimerInstance(ContextDecorator):
    def __init__(self, d, key):
        if key not in d:
            d[key] = list()
        self.d = d
        self.key = key
        self.start_time = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end()

    def start(self):
        self.start_time = time.perf_counter()

    def end(self):
        r"""Manually stop for when using as t = timer(d, k)"""
        if self.start_time is None:
            raise RuntimeError("Timer never initialized")
        self.d[self.key].append(time.perf_counter() - self.start_time)
        self.start_time = None


class Timer:
    #test_n = 1000
    #overhead = None
    #def __init__(self):
    #    times = list()
    #    for i in range(self.test_n):
    #        _data = dict()
    #        timer_instance = self(_data, i)
    #        time0 = time.perf_counter()
    #        timer_instance.start()
    #        timer_instance.end()
    #        times.append(time.perf_counter() - time0)
    #    self.overhead = sum(times) / self.test_n
    #    print(self.overhead)

    def __call__(self, d, key):
        return TimerInstance(d, key)


def get_dense_col_inds(A, row):
    return [col for col in range(A[row].shape[-1]) if A[row, col] != 0]


def get_csr_col_inds(A, row):
    return A.indices[A.indptr[row]:A.indptr[row + 1]]
