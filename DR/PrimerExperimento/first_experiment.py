import logging
import numpy as np
import projections
import umap
import os
from csv import DictWriter
from csv import writer
from scipy import spatial
from sklearn.model_selection import ParameterGrid
from sklearn import (decomposition, manifold)
from openTSNE import (sklearn)

logging.basicConfig(level=logging.DEBUG)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
# Configuration
current_dir = os.getcwd()
dataset = current_dir+"/../../Data/MNIST/mnist_small.npz"
dataset_name = "small_mnist"
output_dir = "Outputs"
max_proj_dim = 200

# Projections with parameters to explore
all_projections = dict()
all_projections['PCA'] = (decomposition.PCA(), {'n_components': [1,2,3,4,5,6,7,8,9,10,50,100,200]})
all_projections['ISOMAP'] = (manifold.Isomap(), {'n_components': [1,2,3,4,5,6,7,8,9,10,50,100,200] , 'n_neighbors': list(range(1,10)), 'eigen_solver': ['dense']})
all_projections['LLE'] = (manifold.LocallyLinearEmbedding(), {'n_components': [1,2,3,4,5,6,7,8,9,10,50,100,200] , 'n_neighbors': list(range(1,10)), 'eigen_solver': ['dense'], 'n_jobs':[-1]})
all_projections['TSNE'] = (sklearn.TSNE(negative_gradient_method="bh"),{'n_components': [1,2,3,4,5,6,7,8,9,10,50,100,200] , 'perplexity':[30,100,250,500], 'exaggeration':[None,2,4]})
all_projections['UMAP'] = (umap.UMAP(),{'n_components': [1,2,3,4,5,6,7,8,9,10,50,100,200] , 'n_neighbors':[5,50,500,1000], 'min_dist':[0,0.1,0.25,0.5,0.75,0.99]})
all_projections['LAPLACIAN-EIGENMAPS'] = (manifold.SpectralEmbedding(), {'n_components': [1,2,3,4,5,6,7,8,9,10,50,100,200] , 'n_neighbors': list(range(1,10)),'n_jobs':[-1]})



##########################################

logging.info("Loading data "+dataset_name)
data = np.load(dataset)
X = data['X']
y = data['y']
nsamples, nx, ny = X.shape
X = X.reshape((nsamples,nx*ny))

# Load or compute X_train distance matrix

try:
    D_high_matrix = np.load('%s/%s_D_high_matrix.npy' % (output_dir, dataset_name))
    D_high_list = np.load('%s/%s_D_high_list.npy' % (output_dir, dataset_name))
    logging.info("Distance loaded from file")
except:
    logging.info("Distance computed and stored")
    D_high_list = spatial.distance.pdist(X, 'euclidean')
    D_high_matrix = spatial.distance.squareform(D_high_list)
    np.save('%s/%s_D_high_list.npy' % (output_dir, dataset_name), D_high_list)
    np.save('%s/%s_D_high_matrix.npy' % (output_dir, dataset_name), D_high_matrix)




for projection_name, proj_tuple in all_projections.items():
    logging.info("Starting evaluation of "+projection_name)
    proj = proj_tuple[0]
    grid_params = ParameterGrid(proj_tuple[1])
    field_names = list(proj_tuple[1].keys())+['trustworthiness', 'continuity', 'normalized_stress', 'neighborhood_hit', 'shepard_diagram_correlation', 'aggregate', 'elapsed_time', 'metrics_time']
    with open(output_dir+"/"+projection_name+".csv","w", newline='') as file:
        filewriter = writer(file, delimiter=',')
        filewriter.writerow(field_names)
    for params in grid_params:
        logging.debug("Evaluating "+projection_name+" with parameters "+str(params))
        proj.set_params(**params)
        X_new, y_new, result = projections.run_projection(proj, X, y, dataset_name, output_dir)
        result = {**params,**result}
        logging.info("Results "+str(result))

        with open(output_dir+"/"+projection_name+".csv","a+", newline='') as file:
            dict_writer = DictWriter(file, fieldnames=field_names)
            dict_writer.writerow(result)
