import subprocess

import numpy as np

from KNN import KNNClassifier
from utils import *

target_attribute = 'Outcome'


def run_knn(k, x_train, y_train, x_test, y_test, formatted_print=True):
    neigh = KNNClassifier(k=k)
    neigh.train(x_train, y_train)
    y_pred = neigh.predict(x_test)
    acc = accuracy(y_test, y_pred)
    print(f'{acc * 100:.2f}%' if formatted_print else acc)


def normalize(x):
    mini = np.min(x, axis=0).reshape((1, 8))
    mini = np.repeat(mini, x.shape[0], axis=0)
    maxi = np.max(x, axis=0).reshape((1, 8))
    maxi = np.repeat(maxi, x.shape[0], axis=0)
    norm_x = (x - mini) / (maxi - mini)
    return norm_x


def power_set(s):
    x = len(s)
    powerset = []
    for i in range(1 << x):
        powerset += [[s[j] for j in range(x) if (i & (1 << j))]]
    return powerset


def farthest_point_sample(x, b):
    top_b_features_indices = []
    corr_dist = 1 - np.corrcoef(x, rowvar=False)
    first = np.argmax(np.mean(corr_dist, axis=1))
    top_b_features_indices.append(first)
    inner_arg = corr_dist[first, :]
    for _ in range(b - 1):
        candidate = inner_arg.argmax()
        top_b_features_indices.append(candidate)
        inner_arg = np.minimum(inner_arg, corr_dist[candidate, :])
    return top_b_features_indices


def get_top_b_features(x, y, b=5, k=51):
    """
    :param k: Number of nearest neighbors.
    :param x: array-like of shape (n_samples, n_features).
    :param y: array-like of shape (n_samples,).
    :param b: number of features to be selected.
    :return: indices of top 'b' features, sorted.
    """
    # TODO: Implement get_top_b_features function
    #   - Note: The brute force approach which examines all subsets of size `b` will not be accepted.

    assert 0 < b < x.shape[1], f'm should be 0 < b <= n_features = {x.shape[1]}; got b={b}.'
    top_b_features_indices = []

    # ====== YOUR CODE: ======
    means = np.mean(x, axis=0)
    class_0 = np.argwhere(y == 0)
    class_1 = np.argwhere(y == 1)
    class_0_means = np.mean(x[class_0, :], axis=0)
    class_1_means = np.mean(x[class_1, :], axis=0)
    class_0_vars = np.var(x[class_0, :], axis=0)
    class_1_vars = np.var(x[class_1, :], axis=0)
    num = class_0.shape[0] * (means - class_0_means) ** 2 + class_1.shape[0] * (means - class_1_means) ** 2
    din = class_0.shape[0] * class_0_vars + class_1.shape[0] * class_1_vars
    top_b_features_indices = np.argsort(num / din)[0].tolist()[::-1]
    top_b_features_indices = top_b_features_indices[:b]
    # ========================

    return top_b_features_indices


def run_cross_validation():
    """
    cross validation experiment, k_choices = [1, 5, 11, 21, 31, 51, 131, 201]
    """
    file_path = str(pathlib.Path(__file__).parent.absolute().joinpath("KNN_CV.pyc"))
    subprocess.run(['python', file_path])


def exp_print(to_print):
    print(to_print + ' ' * (30 - len(to_print)), end='')


# ========================================================================
if __name__ == '__main__':
    """
       Usages helper:
       (*) cross validation experiment
            To run the cross validation experiment over the K,Threshold hyper-parameters
            uncomment below code and run it
    """
    # run_cross_validation()

    # # ========================================================================

    attributes_names, train_dataset, test_dataset = load_data_set('KNN')
    x_train, y_train, x_test, y_test = get_dataset_split(train_set=train_dataset,
                                                         test_set=test_dataset,
                                                         target_attribute='Outcome')

    best_k = 51
    b = 4

    # # ========================================================================

    print("-" * 10 + f'k  = {best_k}' + "-" * 10)
    exp_print('KNN in raw data: ')
    run_knn(best_k, x_train, y_train, x_test, y_test)

    top_m = get_top_b_features(x_train, y_train, b=b, k=best_k)
    x_train_new = x_train[:, top_m]
    x_test_test = x_test[:, top_m]
    exp_print(f'KNN in selected feature data: ')
    run_knn(best_k, x_train_new, y_train, x_test_test, y_test)
