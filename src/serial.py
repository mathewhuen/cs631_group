r"""
"""

import time
from threading import Thread, Event
import multiprocessing
from multiprocessing import Process, Manager, Barrier

import scipy
import numpy as np
from matplotlib import pyplot as plt

from utils import AverageDict, Timer, get_csr_col_inds, get_dense_col_inds


class SerialNetwork:
    def __init__(self, A, SIRN_0, beta, gamma, delta, dt, update_freq, max_steps, output, times):
        self.A = A
        self.n = A.shape[0]
        self.SIRN_0 = SIRN_0
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.dt = dt
        self.update_freq = update_freq
        self.max_steps = max_steps
        self.output = output
        self.times = times
        self.timer = Timer()
        if isinstance(self.A, scipy.sparse.csr_array):
            self.adj = {i: get_csr_col_inds(self.A, i) for i in range(self.n)}
        else:
            self.adj = {i: get_dense_col_inds(self.A, i) for i in range(self.n)}

    def step(self):
        SIRN_1 = self.SIRN_0.copy()
        for i in range(self.n):
            neighbors = self.adj[i]
            _SI = sum([self.A[i, j] * self.SIRN_0[j, 1] / self.SIRN_0[j, 3] for j in neighbors])
            SI = self.beta * self.SIRN_0[i, 0] * _SI
            IR = self.gamma * self.SIRN_0[i, 1]
            RS = self.delta * self.SIRN_0[i, 2]
            dS = RS - SI
            dI = SI - IR
            dR = IR - RS
            SIRN_1[i, 0] += dS * self.dt
            SIRN_1[i, 1] += dI * self.dt
            SIRN_1[i, 2] += dR * self.dt
        self.SIRN_0 = SIRN_1

    def update_output(self, step):
        for i in range(self.n):
            self.output[i][step] = tuple([v.item() for v in self.SIRN_0[i]])

    def run(self):
        times = AverageDict()
        runtime = self.timer(times, "runtime")
        runtime.start()
        for step in range(self.max_steps):
            if self.update_freq is not None and step % self.update_freq == 0:
                with self.timer(times, "update_output"):
                    self.update_output(step)
            with self.timer(times, "step"):
                self.step()
        with self.timer(times, "update_output"):
            self.update_output(step)
        runtime.end()
        self.times[0] = times.calculate()


class SerialManager:
    def __init__(self, A, SIR_0, beta, gamma, delta, dt, max_steps, update_freq=None):
        self.A = A
        self.SIRN_0 = np.hstack([SIR_0, SIR_0.sum(axis=1).reshape(-1, 1)])
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.dt = dt
        self.max_steps = max_steps
        self.update_freq = update_freq

        self.output = {i: dict() for i in range(self.A.shape[0])}
        self.times = dict()

        self.sir_network = SerialNetwork(
            A=self.A,
            SIRN_0=self.SIRN_0,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            dt=self.dt,
            update_freq=self.update_freq,
            max_steps=self.max_steps,
            output=self.output,
            times=self.times,
        )

    def run(self, verbose=True):
        time0 = time.time()
        self.sir_network.run()
        time1 = time.time()
        output_data = self.output
        output_times = self.times
        output = {k: {_k: _v for _k, _v in v.items()} for k, v in output_data.items()}

        time_keys = {
            "runtime": "runtime",
            "update_output": "update_output",
            "step": "update",
        }
        times = dict()
        for proc_id, proc_prof in output_times.items():
            times[proc_id] = dict()
            for k, v in proc_prof.items():
                key = time_keys[k]
                if key not in times[proc_id]:
                    times[proc_id][key] = 0
                times[proc_id][key] += v["total"]

        if verbose:
            print(f"Serial Duration: {time1 - time0:.4f}")

        return {"states": output, "times": times}

    def visualize_basic(self, states=None, times=None):
        from pprint import pprint
        pprint(times)

        fig = plt.figure(figsize=(10, 10))
        for key, data in states.items():
            plt.plot(
                data.keys(), [val[0] / val[3] for val in data.values()],
                label=str(key),
            )
        plt.xlabel("step")
        plt.ylabel("S")
        plt.legend()
        plt.show()

        for node_id in states.keys():
            fig = plt.figure(figsize=(10, 10))
            plt.scatter(
                [val[0] / val[3] for val in states[node_id].values()],
                [val[1] / val[3] for val in states[node_id].values()],
                label=str(node_id),
            )
            plt.xlabel("S")
            plt.ylabel("I")
            plt.legend()
            plt.show()


if __name__ == "__main__":
    import scipy
    multiprocessing.set_start_method("spawn")
    A = np.array([
        [0, .2, 0.7, 0, 0],
        [.1, 0, 0, 0, 0],
        [0.5, 0, 0, 0, 0.1],
        [0, 0, 0, 0, .1],
        [0, 0, 0.075, .1, 0],
    ])
    SIR_0 = np.array([
        [100, 0, 0],
        [100, 1, 0],
        [500, 0, 0],
        [40, 1, 0],
        [100, 1, 0],
    ], dtype=np.float64)
    beta = 0.35
    gamma = 0.1
    delta = 0.01
    dt = 0.2
    max_steps = 500
    serial_manager = SerialManager(
        A=A,
        SIR_0=SIR_0,
        beta=beta,
        gamma=gamma,
        delta=delta,
        dt=dt,
        max_steps=max_steps,
    )
    data = serial_manager.run()
    # from pprint import pprint
    # pprint(data["states"][0])
