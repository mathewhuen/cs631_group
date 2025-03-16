import multiprocessing

from network import generate_data
from partition import spectral_partition
from parallel import ParallelManager
from serial import SerialManager

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
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
