import time

import numpy as np
from numpy import ndarray
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC

from hw1 import train_labels, train_data, test_data, test_labels, train_data_edges, test_data_edges, label_names


def time_function(f, *args, **kwargs):
    start = time.time()
    ret = f(*args, **kwargs)
    end = time.time()
    if True:
        print(end - start)
    return ret

train_data = np.concatenate(
    (train_data, train_data_edges.reshape(50000, 1, 32, 32)),
    axis=1,
)
test_data = np.concatenate(
    (test_data, test_data_edges.reshape(10000, 1, 32, 32)),
    axis=1,
)

flatten_features = FunctionTransformer(lambda x: x.reshape(x.shape[0], -1))

clf = make_pipeline(
    flatten_features,
    StandardScaler(),
    PCA(n_components=50),
    SVC(),
)

sample_range = slice(0, 50000)  # Use a smaller sample range for faster execution

time_function(clf.fit, train_data[sample_range], train_labels[sample_range])
y_pred = time_function(clf.predict, test_data)
print(metrics.classification_report(test_labels, y_pred, digits=4, target_names=label_names))

cm = metrics.confusion_matrix(test_labels, y_pred)
print(cm)
