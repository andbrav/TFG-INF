import logging
import numpy as np
from time import perf_counter
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from scipy import spatial

def trustworthiness(k, D_high, D_low):
    logging.info("Computing trustworthiness")

    n = D_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    # knn_orig[i] son los k-vecinos de i en proyección
    knn_orig = nn_orig[:, :k + 1][:, 1:]
    # knn_proj[i] son los k-vecinos de i en proyección
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())



def continuity(k, D_high, D_low):
    logging.info("Computing continuity")

    n = D_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    # knn_orig[i] son los k-vecinos de i en proyección
    knn_orig = nn_orig[:, :k + 1][:, 1:]
    # knn_proj[i] son los k-vecinos de i en proyección
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


def normalized_stress(D_high,D_low):
    logging.info("Computing normalized stress")

    return np.sum((D_high - D_low)**2) / np.sum(D_high**2)

def neighborhood_hit(X, y, k):
    # X is data, y labels and k number of neighbors
    logging.info("Computing neighborhood hit")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    neighbors = knn.kneighbors(X, return_distance=False)
    return np.mean(np.mean((y[neighbors] == np.tile(y.reshape((-1, 1)), k)).astype('uint8'), axis=1))

def shepard_diagram_correlation(D_high, D_low):
    logging.info("Computing shepard diagram correlation")
    return stats.spearmanr(D_high, D_low)[0]



def empty_metrics():
    results = dict()

    results['trustworthiness'] = 0
    results['continuity'] = 0
    results['normalized_stress'] = 0
    results['neighborhood_hit'] = 0
    results['shepard_diagram_correlation'] = 0
    results['aggregate'] = 0
    results['elapsed_time'] =  0

    return results


def eval_pq_metrics(**kwargs):

    X = kwargs.get('X', None)
    y = kwargs.get('y', None)
    dataset_name = kwargs.get('dataset_name', None)
    output_dir = kwargs.get('output_dir', None)

    D_low_list = spatial.distance.pdist(X, 'euclidean')
    D_low_matrix = spatial.distance.squareform(D_low_list)

    np.save("%s/%s_D_low_matrix.npy" % (output_dir, dataset_name), D_low_matrix)
    np.save("%s/%s_D_low_list.npy" % (output_dir, dataset_name), D_low_list)

    #D_low_list = None
    #D_low_matrix = None

    D_high = np.load("%s/%s_D_high_matrix.npy" % (output_dir, dataset_name), mmap_mode='c')
    D_low = np.load("%s/%s_D_low_matrix.npy" % (output_dir, dataset_name), mmap_mode='c')
    D_high_list = np.load("%s/%s_D_high_list.npy" % (output_dir, dataset_name), mmap_mode='c')
    D_low_list = np.load("%s/%s_D_low_list.npy" % (output_dir, dataset_name), mmap_mode='c')

    results = dict()
    t0 = perf_counter()

    results['trustworthiness'] = trustworthiness(7, D_high, D_low)
    results['continuity'] = continuity(7, D_high, D_low)
    results['normalized_stress'] = normalized_stress(D_high_list,D_low_list)
    stress = results['normalized_stress'] if results['normalized_stress']<=1 else 1
    results['neighborhood_hit'] = neighborhood_hit(X, y, 7)
    results['shepard_diagram_correlation'] = shepard_diagram_correlation(D_high_list, D_low_list)
    correlation = results['shepard_diagram_correlation'] if results['shepard_diagram_correlation']>=0 else 0
    results['aggregate'] = (1/5)*(results['trustworthiness']+results['continuity']+(1-stress)+results['neighborhood_hit']+correlation)
    results['elapsed_time'] = kwargs.get('elapsed_time', 0.0)
    metrics_time = perf_counter() - t0
    results['metrics_time'] = metrics_time

    return results
