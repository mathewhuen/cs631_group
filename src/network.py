import time
import scipy
import numpy as np
import networkx as nx
import matplotlib.cm as cm
from matplotlib import pyplot as plt


MAX_DIST = np.pow(2, 0.5)


def weight(a, b):
    return (MAX_DIST - np.pow(np.pow((a - b), 2).sum(), 0.5)) / MAX_DIST


def sample_grid(n_nodes, n_hubs=0, suburb_factor=10):
    # sample from square, add hubs, connect as a grid.
    points = np.random.uniform(size=(n_nodes, 2))
    if n_hubs > 0:
        hubs = np.random.uniform(size=(n_hubs, 2))
        suburbs = list()
        for hub in hubs:
            for _ in range(suburb_factor):
                suburb = np.array([
                    scipy.stats.truncnorm.rvs(0, 1, coord, 0.02)
                    for coord in hub
                ])
                suburbs.append(suburb)

    points = np.vstack([points, hubs] + [suburbs])
    n = len(points)
    triangulation = scipy.spatial.Delaunay(points)
    A = scipy.sparse.csr_array((n, n))
    for simplex in triangulation.simplices:
        for i in simplex:
            for j in simplex:
                if i != j and A[i, j] == 0:
                    A[i, j] = weight(points[i], points[j])
    return A, points


def sample_SIRN_random(A, min_N, max_N):
    n = A.shape[0]
    SIRN = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        SIRN[i, :-1] = np.random.choice(range(int(min_N / 3), int(max_N / 3)), size=(3,))
        SIRN[i, -1] = SIRN[i].sum()
    return SIRN


def sample_SIRN_1n(A, min_N, max_N):
    n = A.shape[0]
    SIRN = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        SIRN[i, [0, 2]] = np.random.choice(range(int(min_N / 3), int(max_N / 3)), size=(2,))
        SIRN[i, 1] = 1
        SIRN[i, -1] = SIRN[i].sum()
    return SIRN


def sample_SIRN_weighted(A, min_N, max_N):
    n = A.shape[0]
    SIRN = np.zeros((n, 4), dtype=np.float64)

    ws = A.sum(axis=-1)
    ws = ws / ws.max()

    for i in range(n):
        # m = len(get_col_inds(A, i))
        N = int(ws[i] * (max_N - min_N) + min_N)
        S = np.random.choice(range(N))
        I = np.random.choice(range(N - S))
        SIRN[i, 0] = S
        SIRN[i, 1] = I
        SIRN[i, 2] = N - (S + I)
        SIRN[i, 3] = N
    return SIRN


def generate_data(
    n_nodes,
    n_hubs,
    suburb_factor,
    min_N,
    max_N,
    SIRN_strategy="random",
):
    A, points = sample_grid(n_nodes, n_hubs=n_hubs, suburb_factor=suburb_factor)
    if SIRN_strategy == "random":
        SIRN = sample_SIRN_random(A, min_N, max_N)
    elif SIRN_strategy == "1_infected_per":
        SIRN = sample_SIRN_1n(A, min_N, max_N)
    elif SIRN_strategy == "weighted":
        SIRN = sample_SIRN_weighted(A, min_N, max_N)
    else:
        raise NotImplementedError()
    return A, points, SIRN


def visualize_partitions(A, partitions, points=None, node_size=75):
    G = nx.from_scipy_sparse_array(A)
    plt.figure(figsize=(10, 10))
    unique_partitions = list(partitions.keys())
    cmap = cm.get_cmap("tab10", len(unique_partitions))
    node_colors = {}
    for cluster_id, node_list in partitions.items():
        color = cmap(cluster_id)  # Assign a color to the cluster
        for node in node_list:
            node_colors[node] = color
    if points is not None:
        positions = {row_i: row for row_i, row in enumerate(points)}
    else:
        positions = None
    nx.draw(
        G,
        positions,
        with_labels=False,
        node_color=[node_colors[n] for n in G.nodes()],
        edge_color="gray",
        node_size=node_size,
        cmap=cmap,
    )
    plt.show()


def SIRN_color(SIRN):
    return (SIRN[1] / SIRN[3], 0.2, 0.4, 0.9)


def visualize_SIRN(A, SIRN, points=None, node_size=75):
    G = nx.from_scipy_sparse_array(A)
    plt.figure(figsize=(10, 10))
    node_colors = [SIRN_color(SIRN[node_id]) for node_id in G.nodes()]
    if points is not None:
        positions = {row_i: row for row_i, row in enumerate(points)}
    else:
        positions = None
    nx.draw(
        G,
        positions,
        with_labels=False,
        node_color=node_colors,
        edge_color="gray",
        node_size=node_size,
    )
    plt.show()


def visualize_SIRNs(A, SIRNs, points=None, node_size=75):
    G = nx.from_scipy_sparse_array(A)
    fig, ax = plt.subplots(figsize=(10, 10))
    node_colors = [SIRN_color(SIRNs[0][node_id]) for node_id in G.nodes()]
    if points is not None:
        positions = {row_i: row for row_i, row in enumerate(points)}
    else:
        positions = nx.spring_layout(G)
    nodes = nx.draw_networkx_nodes(
        G,
        positions,
        node_color=node_colors,
        node_size=node_size,
        ax=ax,
    )
    edges = nx.draw_networkx_edges(
        G,
        positions,
        edge_color="gray",
        ax=ax,
    )
    plt.draw()
    plt.pause(0.1)
    for step in range(1, len(SIRNs)):
        node_colors = [SIRN_color(SIRNs[step][node_id]) for node_id in G.nodes()]
        nodes.set_facecolor(node_colors)
        plt.draw()
        plt.pause(0.1)


if __name__ == "__main__":
    from partition import spectral_partition
    from parallel import ParallelManager as PM

    A, points, SIRN = generate_data(100, 5, 10, 100, 200, SIRN_strategy="random")  # SIRN_strategy="1_infected_per")
    partitions = spectral_partition(A, n_workers=1, levels=2)
    visualize_partitions(A, partitions, points=points)
    visualize_SIRN(A, SIRN, points=points)
