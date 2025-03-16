r"""
Parallel Epidemiological Simulator.

Call :class:`ParallelManager` to run simulations as follows:
```python
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
])
assignments = [(0, 1, 2), (3, 4)]
beta = 0.35
gamma = 0.1
delta = 0.01
dt = 0.2
update_freq = 10
max_steps = 500
parallel_manager = ParallelManager(
    A=A,
    SIR_0=SIR_0,
    assignments=assignments,
    beta=beta,
    gamma=gamma,
    delta=delta,
    dt=dt,
    max_steps=max_steps,
    update_freq=update_freq,
)
data = parallel_manager.run()
```
"""

import time
from threading import Thread, Event
import multiprocessing
from multiprocessing import Process, Manager, Barrier, Queue
import queue

import scipy
import numpy as np
from matplotlib import pyplot as plt

from utils import AverageDict, Timer, get_csr_col_inds, get_dense_col_inds


def update_shared_data(namespace, share_node_ids, nodes, step):
    for node_id in share_node_ids:
        namespace.share_vals[node_id] = {
            "S": nodes[node_id].S,
            "I": nodes[node_id].I,
            "R": nodes[node_id].R,
            "N": nodes[node_id].N,
            "step": step,
            "id": node_id,
        }


class DataCollector(Thread):
    def __init__(self, q, timeout=0.2):
        super().__init__()
        self.q = q
        self.timeout = timeout
        self.output = dict()

    def process_data(self, data):
        node_id, step, SIRN = data
        if node_id not in self.output:
            self.output[node_id] = dict()
        self.output[node_id][step] = SIRN

    def run(self):
        while True:
            try:
                data = self.q.get(timeout=self.timeout)
                if data is None:
                    break
                self.process_data(data)
            except queue.Empty:
                pass


class Neighborhood(Process):
    def __init__(
        self,
        proc_id, A, SIR_0, ids, namespace, barrier, beta, gamma, delta, sync_freq,
        dt=1, update_freq=None, max_steps=1_000, output_stream=None,
    ):
        super().__init__()

        self.proc_id = proc_id
        get_col_inds = get_csr_col_inds if isinstance(A, scipy.sparse.csr_array) else get_dense_col_inds
        self.nodes = {
            int(idx): Node(
                node_id=idx,
                # adj={j: A[idx, j] for j in range(A[idx].shape[-1]) if A[idx, j] != 0},
                adj={int(j): A[idx, j] for j in get_col_inds(A, idx)},
                S_0=SIR_0[idx, 0],
                I_0=SIR_0[idx, 1],
                R_0=SIR_0[idx, 2],
                beta=beta,
                gamma=gamma,
                delta=delta,
                dt=dt,
            )
            for idx in ids
        }

        self.internal_node_ids = list(self.nodes.keys())
        self.external_node_ids = list(set([
            k
            for node in self.nodes.values()
            for k in node.adj.keys()
            if k not in self.internal_node_ids
        ]))
        self.share_node_ids = list(set([
            node_id
            for node_id, node in self.nodes.items()
            for k in node.adj.keys()
            if k not in self.internal_node_ids
        ]))

        for node_id in self.share_node_ids:
            namespace.share_vals[node_id] = {"S": None, "I": None, "R": None, "N": None, "step": -100, "id": node_id}

        self.sync_freq = sync_freq
        self.namespace = namespace
        self.update_freq = update_freq
        self.max_steps = max_steps
        self.barrier = barrier
        self.output_stream = output_stream

        self.timer = Timer()

    def update_shared_data(self, step):
        for node_id in self.share_node_ids:
            self.namespace.share_vals[node_id] = {
                "S": self.nodes[node_id].S,
                "I": self.nodes[node_id].I,
                "R": self.nodes[node_id].R,
                "N": self.nodes[node_id].N,
                "step": step,
                "id": node_id,
            }

    def get_val(self, node_id, step):
        # node_id = int(node_id)
        if node_id in self.nodes:
            node = self.nodes[node_id]
            return {"S": node.S, "I": node.I, "R": node.R, "N": node.N, "step": step, "id": node_id}, True
        if step - self.namespace.share_vals[node_id]["step"] >= self.sync_freq:
            return self.namespace.share_vals[node_id], False
        return self.namespace.share_vals[node_id], True

    def update_internal_nodes(self, step):
        for node_id, node in self.nodes.items():
            for j in node.adj.keys():
                vals, resolved = self.get_val(j, step)
                if resolved:
                    node.update(vals)
                else:
                    # node.queue(vals)
                    node.queue((self.namespace.share_vals, j))

    def finalize_internal_nodes(self):
        for node_id, node in self.nodes.items():
            node.resolve()

    def update_output(self, step):
        for node_id, node in self.nodes.items():
            if self.output_stream:
                self.output_stream.put((node_id, step, (node.S.item(), node.I.item(), node.R.item(), node.N.item())))
            else:
                self.namespace.output[node_id][step] = (node.S.item(), node.I.item(), node.R.item(), node.N.item())

    def run(self):
        times = AverageDict()
        runtime = self.timer(times, "runtime")
        runtime.start()
        steptime = self.timer(times, "steptime")
        for step in range(self.max_steps):
            steptime.start()

            if self.update_freq is not None and step % self.update_freq == 0:
                with self.timer(times, "update_output"):
                    self.update_output(step)

            if step % self.sync_freq == 0:
                with self.timer(times, "neighborhood_sync_1"):
                    self.barrier.wait()

            if step % self.sync_freq == 0:
                with self.timer(times, "thread_init"):
                    t0 = Thread(
                        target=update_shared_data,
                        args=(self.namespace, self.share_node_ids, self.nodes, step),
                    )
                    t0.start()

            with self.timer(times, "update_internal_1"):
                self.update_internal_nodes(step)

            # wait for internal nodes to finish
            if step % self.sync_freq == 0:
                with self.timer(times, "external_node_load_sync"):
                    t0.join()
                    self.barrier.wait()
            with self.timer(times, "update_internal_2"):
                self.finalize_internal_nodes()

            # wait for all neighborhoods
            if step % self.sync_freq == 0:
                with self.timer(times, "neighborhood_sync_2"):
                    self.barrier.wait()

            steptime.end()

        with self.timer(times, "update_output"):
            self.update_output(step)
        runtime.end()
        self.namespace.times[self.proc_id] = times.calculate()


class Node:
    def __init__(self, node_id, adj, S_0, I_0, R_0, beta, gamma, delta, dt=1):
        super().__init__()
        self.node_id = node_id
        self.adj = adj
        self.S = S_0
        self.I = I_0
        self.R = R_0
        self.N = S_0 + I_0 + R_0
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.dt = dt

        self._SI = 0
        self._queue = list()

    def update(self, d):
        self._SI += self.adj[d["id"]] * d["I"] / d["N"]

    def queue(self, d):
        self._queue.append(d)

    def resolve(self):
        for share_vals, node_id in self._queue:
            item = share_vals[node_id]
            self._SI += self.adj[item["id"]] * item["I"] / item["N"]
        SI = self.beta * self.S * self._SI
        IR = self.gamma * self.I
        RS = self.delta * self.R
        dS = RS - SI
        dI = SI - IR
        dR = IR - RS
        self.S += dS * self.dt
        self.I += dI * self.dt
        self.R += dR * self.dt

        self._SI = 0
        self._queue = list()


class ParallelManager:
    def __init__(
        self, A, SIR_0, assignments, beta, gamma, delta, dt, max_steps,
        sync_freq=10, update_freq=None, use_data_streaming=True,
    ):
        self.A = A
        self.SIR_0 = SIR_0
        self.assignments = assignments
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.dt = dt
        self.max_steps = max_steps

        self.barrier = Barrier(len(self.assignments))
        self.manager = Manager()
        self.namespace = self.manager.Namespace()
        self.namespace.share_vals = self.manager.dict()
        self.namespace.output = self.manager.dict()
        self.namespace.times = self.manager.dict()
        for i in range(A.shape[0]):
            self.namespace.output[i] = self.manager.dict()
        self.sync_freq = sync_freq
        self.update_freq = update_freq  # update the output dataset. Use None if not streaming

        self.use_data_streaming = use_data_streaming
        if use_data_streaming:
            self.data_queue = Queue()
            self.data_collector = DataCollector(self.data_queue)
        else:
            self.data_queue = None
            self.data_collector = None

        self.neighborhoods = [
            Neighborhood(
                proc_id=proc_id,
                A=self.A,
                SIR_0=self.SIR_0,
                ids=assignment,
                namespace=self.namespace,
                barrier=self.barrier,
                beta=self.beta,
                gamma=self.gamma,
                delta=self.delta,
                sync_freq=self.sync_freq,
                dt=self.dt,
                update_freq=self.update_freq,
                max_steps=self.max_steps,
                output_stream=self.data_queue,
            )
            for proc_id, assignment in enumerate(self.assignments)
        ]

    def run(self, verbose=True):
        if self.use_data_streaming:
            self.data_collector.start()
        time0 = time.time()
        for neighborhood in self.neighborhoods:
            neighborhood.start()
        for neighborhood in self.neighborhoods:
            neighborhood.join()
        time1 = time.time()

        if self.use_data_streaming:
            self.data_queue.put(None)
            self.data_collector.join()
            output = self.data_collector.output
            output = {k: {_k: _v for _k, _v in v.items()} for k, v in output.items()}
        else:
            output = {k: {_k: _v for _k, _v in v.items()} for k, v in self.namespace.output.items()}

        time_keys = {
            "runtime": "runtime",
            "steptime": "steptime",
            "update_output": "update_output",
            "neighborhood_sync_1": "sync",
            "neighborhood_sync_2": "sync",
            "thread_init": "thread_init",
            "update_internal_1": "update",
            "update_internal_2": "update",
            "external_node_load_sync": "non_local_load",
        }
        times = dict()
        for proc_id, proc_prof in self.namespace.times.items():
            times[proc_id] = dict()
            for k, v in proc_prof.items():
                key = time_keys[k]
                if key not in times[proc_id]:
                    times[proc_id][key] = 0
                times[proc_id][key] += v["total"]

        if verbose:
            print(f"Loose Sync (Every {self.sync_freq}) Duration: {time1 - time0:.4f}")

        return {"states": output, "times": times}

    def visualize_basic(self, states=None, times=None):
        from pprint import pprint
        if times is not None:
            pprint(times)

        if states is not None:
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
    ])
    assignments = [(0, 1, 2), (3, 4)]
    beta = 0.35
    gamma = 0.1
    delta = 0.01
    dt = 0.2
    update_freq = None
    max_steps = 500
    use_data_streaming = True
    parallel_manager = ParallelManager(
        A=A,
        SIR_0=SIR_0,
        assignments=assignments,
        beta=beta,
        gamma=gamma,
        delta=delta,
        dt=dt,
        max_steps=max_steps,
        update_freq=update_freq,
        use_data_streaming=use_data_streaming,
    )
    data = parallel_manager.run()
    # parallel_manager.visualize_basic(**data)
    # from pprint import pprint
    # pprint(data["states"][0])
