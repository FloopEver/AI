import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    real = np.array(real_labels)
    predicted = np.array(predicted_labels)
    TP = sum(real * predicted)
    precision = TP / sum(predicted) if sum(predicted) else 0
    recall = TP / sum(real) if sum(real) else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return f1_score


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        d = np.cbrt((abs(p1 - p2) ** 3).sum())
        return d

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        d = pow(((p1 - p2) ** 2).sum(), 1 / 2)
        return d

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        p1 = np.array(point1)
        p2 = np.array(point2)
        p1d = pow((p1 ** 2).sum(), 0.5)
        p2d = pow((p2 ** 2).sum(), 0.5)
        if p1d == 0 or p2d == 0:
            d = 1
        else:
            d = 1 - ((p1 * p2).sum() / (p1d * p2d))

        return d

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        a = 'euclidean'
        b = 'minkowski'
        c = 'cosine_dist'
        re = [0, 0, 'euclidean', None]
        for ch in [a, b, c]:
            for i in range(1, 31, 2):
                knn_instance = KNN(i, distance_funcs[ch])
                knn_instance.train(x_train, y_train)
                pred = knn_instance.predict(x_val)
                f1score = f1_score(y_val, pred)
                if f1score > re[0]:
                    re = f1score, i, ch, knn_instance
        self.best_k = re[1]
        self.best_distance_function = re[2]
        self.best_model = re[3]


    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        a = 'euclidean'
        b = 'minkowski'
        c = 'cosine_dist'
        re = [0, 0, 'euclidean', None, 'min_max_scale']
        normalize = ['min_max_scale', 'normalize']
        for r in range(2):
            nc = scaling_classes[normalize[r]]()
            x_train_n = nc(x_train)
            x_val_n = nc(x_val)
            for ch in [a, b, c]:
                for i in range(1, 31, 2):
                    knn_instance = KNN(i, distance_funcs[ch])
                    knn_instance.train(x_train_n, y_train)
                    pred = knn_instance.predict(x_val_n)
                    f1score = f1_score(y_val, pred)
                    if f1score > re[0]:
                        re = f1score, i, ch, knn_instance, normalize[r]
        self.best_k = re[1]
        self.best_distance_function = re[2]
        self.best_model = re[3]
        self.best_scaler = re[4]

class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normarlize_feature = []
        for feature in features:
            n_feature = []
            f = np.array(feature)
            f_2 = pow((f ** 2).sum(), 0.5)
            for i in feature:
                if f_2 == 0:
                    n_feature.append(0)
                else:
                    n_feature.append(i / f_2)
            normarlize_feature.append(n_feature)
        return normarlize_feature

class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        f = np.array(features)
        normarlize_feature = []
        max_f = f.max(axis=0)
        min_f = f.min(axis=0)
        for feature in f:
            n_f = []
            for i in range(len(feature)):
                n_i = (feature[i] - min_f[i]) / (max_f[i] - min_f[i]) if (max_f[i] - min_f[i]) else 0
                n_f.append(n_i)
            normarlize_feature.append(n_f)
        return normarlize_feature

