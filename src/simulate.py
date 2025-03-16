import json
import time
import argparse
import multiprocessing
from pathlib import Path
from pprint import pprint

import scipy
import numpy as np

from network import generate_data
from partition import spectral_partition
from parallel import ParallelManager
from serial import SerialManager
from grp_network import generate_random_data


def run_example():
    from pprint import pprint

    beta = 0.35
    gamma = 0.1
    delta = 0.01
    dt = 0.2
    max_steps = 15

    n_regular_nodes = 10_000
    n_hubs = 200
    suburb_factor = 15
    partition_levels = 2
    #n_regular_nodes = 100
    #n_hubs = 20
    #suburb_factor = 5
    #partition_levels = 1

    min_N = 100
    max_N = 200
    SIRN_strategy = "random"
    # SIRN_strategy = "1_infected_per"

    n_preprocessing_workers = 1

    A, points, SIRN = generate_data(
        n_regular_nodes,
        n_hubs=n_hubs,
        suburb_factor=suburb_factor,
        min_N=min_N,
        max_N=max_N,
        SIRN_strategy=SIRN_strategy,
    )
    # SIR = sample_SIRN_random(A, min_N, max_N)[:, :-1]
    SIR = SIRN[:, :-1]
    assignments = [
        tuple(inds)
        for inds in spectral_partition(
            A,
            n_workers=n_preprocessing_workers,
            levels=partition_levels,
        ).values()
    ]
    print(f"{len(assignments)} processes")

    parallel_manager = ParallelManager(
        A=A,
        SIR_0=SIR,
        assignments=assignments,
        beta=beta,
        gamma=gamma,
        delta=delta,
        dt=dt,
        max_steps=max_steps,
    )
    parallel_data = parallel_manager.run()
    pprint(parallel_data["states"][0])
    serial_manager = SerialManager(
        A=A,
        SIR_0=SIR,
        beta=beta,
        gamma=gamma,
        delta=delta,
        dt=dt,
        max_steps=max_steps,
    )
    serial_data = serial_manager.run()
    pprint(serial_data["states"][0])


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--serial", "-s", action="store_true")
    parser.add_argument("--min_N", type=int, required=True)
    parser.add_argument("--max_N", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--update_freq", type=int, required=True)
    parser.add_argument("--n_regular_nodes", type=int, required=True)
    parser.add_argument("--n_hubs", type=int, required=True)
    parser.add_argument("--suburb_factor", type=int, required=True)
    parser.add_argument("--partition_levels", type=int, required=True)
    parser.add_argument("--SIRN_strategy", type=str, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--network_load_scheme", type=str, default="map")  # map, random
    parser.add_argument("--use_data_streaming", action="store_true")
    parser.add_argument("--data_load_path", type=str)
    parser.add_argument("--data_save_path", type=str)

    return vars(parser.parse_args())


def run_experiment(
    serial,
    min_N, max_N,
    beta, gamma, delta, dt,
    max_steps,
    update_freq,
    n_regular_nodes, n_hubs, suburb_factor,
    partition_levels, SIRN_strategy, n_workers, network_load_scheme, use_data_streaming,
    data_load_path, data_save_path,
    params,  # hacky: for storing all of the args together
):
    timestamp = time.time()
    print(f"Timestamp: {timestamp}")

    if data_load_path is not None:
        data_load_path = Path(data_load_path)
    if data_save_path is not None:
        data_save_path = Path(data_save_path) / str(timestamp)
        if data_save_path.exists() and not data_save_path.is_dir():
            raise RuntimeError("A file already exists at the expected output path: {}".format(
                data_save_path.as_posix(),
            ))
        if not data_save_path.exists():
            data_save_path.mkdir(parents=True)

    # data
    if data_load_path is None:
        if network_load_scheme == "map":
            A, points, SIRN = generate_data(
                n_regular_nodes,
                n_hubs=n_hubs,
                suburb_factor=suburb_factor,
                min_N=min_N,
                max_N=max_N,
                SIRN_strategy=SIRN_strategy,
            )
        else:
            _n_nodes = n_regular_nodes+n_hubs*(suburb_factor+1)
            A, SIRN = generate_random_data(
                node_range=(_n_nodes, _n_nodes+1),
                sparsity_range=(0.1, 0.3),
                mixing_rate_range=(0.1, 0.9),
                population_range=(min_N, max_N),
            )
            points = None
    else:
        A = scipy.sparse.load_npz(data_load_path / "A.npz")
        SIRN = np.load(data_load_path / "SIRN.npy")
        if (data_load_path / "points.npy").exists():
            points = np.load(data_load_path / "points.npy")
        else:
            points = None

    SIR = SIRN[:, :-1]

    # experiment
    if serial:
        serial_manager = SerialManager(
            A=A,
            SIR_0=SIR,
            beta=beta,
            gamma=gamma,
            delta=delta,
            dt=dt,
            max_steps=max_steps,
            update_freq=update_freq,
        )
        data = serial_manager.run()
    else:
        assignments = [
            tuple(inds)
            for inds in spectral_partition(
                A, n_workers=n_workers, levels=partition_levels,
            ).values()
        ]
        parallel_manager = ParallelManager(
            A=A,
            SIR_0=SIR,
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

    if data_save_path is None:
        pprint(data["states"][0])
    else:
        with open(data_save_path / "states.json", "w") as f:
            json.dump(data["states"], f, indent=2)
        with open(data_save_path / "times.json", "w") as f:
            json.dump(data["times"], f, indent=2)
        with open(data_save_path / "params.json", "w") as f:
            json.dump(params, f, indent=2)
        scipy.sparse.save_npz(data_save_path / "A.npz", A)
        np.save(data_save_path / "SIRN.npy", SIRN)
        if points is not None:
            np.save(data_save_path / "points.npy", points)
        if data_load_path is not None:
            with open(data_save_path / "continued.json", "w") as f:
                json.dump({"from": data_load_path.as_posix()}, f, indent=2)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    params = parse_arguments()
    run_experiment(params=params, **params)
