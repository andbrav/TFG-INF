import logging
import sys
import numpy as np
from sklearn import (decomposition, manifold)
from time import perf_counter
import metrics

logging.basicConfig(level=logging.DEBUG)

def run_projection(proj, X, y, dataset_name, output_dir):
    t0 = perf_counter()
    number_dimensions = proj.get_params()['n_components']
    #try:
    X_new = proj.fit_transform(X, y)
    #except:
     #   logging.error("Running %s" % str(proj.get_params()))
      #  reason, _, tb = sys.exc_info()
       # logging.error('Reason: '+str(reason))
        #return np.zeros((X.shape[0], number_dimensions)), y, metrics.empty_metrics()

    elapsed_time = perf_counter() - t0

    if X_new.shape[0] != X.shape[0]:
        logging.error("Running %s: Projection returned %d rows when %d rows were expected" % (str(proj.get_params()), X_new.shape[0], X.shape[0]))
        return np.zeros((X.shape[0], number_dimensions)), y, metrics.empty_metrics()

    if len(X_new.shape) != 2 or X_new.shape[1] != number_dimensions:
        logging.error("Error running %s: Projection returned %d dimensions when %d were expected " % (str(proj.get_params()),X_new.shape[1],number_dimensions))
        return np.zeros((X.shape[0], number_dimensions)), y, metrics.empty_metrics()

    return X_new, y, metrics.eval_pq_metrics(X=X_new, y=y, elapsed_time=elapsed_time, dataset_name=dataset_name, output_dir=output_dir)
