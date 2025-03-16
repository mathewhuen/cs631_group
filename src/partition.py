import scipy
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def spectral_partition_step(A, inds):
    if A.shape[0] <= 3:
        return [inds]
    L = scipy.sparse.csr_matrix(scipy.sparse.csgraph.laplacian(A, normed=True))
    eigvals, eigvecs = scipy.sparse.linalg.eigs(L, k=2, which="SR")  # ordered in decreasing magnitude
    clusters = np.sign(eigvecs[:, eigvals.argmax()])
    left_is = np.where(clusters <= 0)[0]
    right_is = np.where(clusters > 0)[0]

    if len(left_is) == 0 or len(right_is) == 0:
        return [inds]

    for sub_inds in (left_is, right_is):
        subgraph = A[sub_inds, :][:, sub_inds]
        n_comps, _ = scipy.sparse.csgraph.connected_components(subgraph)
        if n_comps > 1:
            return [inds]
    return [ [inds[i] for i in left_is], [inds[i] for i in right_is] ]


def spectral_partition_sp(A, levels=2):
    clusters = spectral_partition_step(A, list(range(A.shape[0])))
    for level in range(levels):
        new_clusters = list()
        for inds in clusters:
            new_clusters.extend(spectral_partition_step(A[inds, :][:, inds], inds))
        clusters = new_clusters
    return clusters


def spectral_partition_mp(A, n_workers=-1, levels=2):
    clusters = spectral_partition_step(A, list(range(A.shape[0])))
    with ProcessPoolExecutor(n_workers) as executor:
        for _ in range(levels - 1):
            futures = [executor.submit(spectral_partition_step, A[inds, :][:, inds], inds) for inds in clusters]
            clusters = list()
            for future in futures:
                clusters.extend(future.result())
    return clusters


def spectral_partition(A, n_workers=-1, levels=2):
    if n_workers == 1:
        clusters = spectral_partition_sp(A, levels=levels)
    else:
        clusters = spectral_partition_mp(A, n_workers=n_workers, levels=levels)

    output = dict()
    for cluster_i, cluster in enumerate(clusters):
        output[cluster_i] = cluster
    return output


if __name__ == "__main__":
    import time
    n = 1000
    A = np.random.choice(range(10), size=(n, n))
    time0 = time.perf_counter()
    partitions = spectral_partition(A, n_workers=1, levels=4)
    print(time.perf_counter() - time0)
    print("partition sizes: {}".format({k: len(v) for k, v in partitions.items()}))
