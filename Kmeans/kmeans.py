import numpy as np


#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''

    ###############################################
    # TODO: implement the Kmeans++ initialization
    ###############################################
    # L2 distance
    def distance(x1, x2):
        return np.sum(np.square(x1 - x2))

    # find min dist for one point and selected centers
    def nearest(point, cluster_centers):
        min_dist = np.inf
        m = np.shape(cluster_centers)[0]
        for i in range(m):
            d = distance(point, x[int(cluster_centers[i])])
            if min_dist >= d:
                min_dist = d
        return min_dist

    centers = np.zeros(n_cluster)
    centers[0] = generator.randint(0, n)
    d = [0.0 for _ in range(n)]
    for i in range(1, n_cluster):
        s = 0
        for j in range(n):
            d[j] = nearest(x[j], centers[0:i])
            s += d[j]
        s *= generator.rand()
        for j, di in enumerate(d):
            s -= di
            if s > 0:
                continue
            centers[i] = j
            break
    list_centers = []
    for i in centers:
        list_centers.append(int(i))
    # DO NOT CHANGE CODE BELOW THIS LINE
    print(list_centers)
    return list_centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array,
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0),
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        centroids = np.zeros((self.n_cluster, D))
        for i in range(self.n_cluster):
            centroids[i] = x[int(self.centers[i])]
        membership = np.zeros(N, dtype=int)
        objective = np.inf
        step = 0

        for step in range(self.max_iter):
            distance = np.empty((0, N), float)
            new_objective = 0
            for i in range(self.n_cluster):
                l2 = np.sum((centroids[i] - x) ** 2, axis=1)
                distance = np.append(distance, [l2], axis=0)
            membership = np.argmin(distance, axis=0)
            for i in range(self.n_cluster):
                new_objective += np.sum(distance[i, :] * np.array(membership == i))

            new_objective /= N

            # average K-mean objective changes less than self.e
            if abs(objective - new_objective) <= self.e:
                break

            objective = new_objective
            for k in range(self.n_cluster):
                point_in_cluster = np.array(membership == k)
                centroids[k] = np.dot(point_in_cluster, x)
                count = np.count_nonzero(membership == k)
                if count > 0:
                    centroids[k] = centroids[k] / count
            step += 1
        return (centroids, membership, step)


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented,
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        centroids, membership, step = KMeans(self.n_cluster, max_iter=self.max_iter, e=self.e,
                                             generator=self.generator).fit(x, centroid_func)
        centroid_labels = np.empty(0, )

        # Training
        for k in range(self.n_cluster):
            vote = np.empty((0,))
            rk = np.array(membership == k)
            if np.count_nonzero(rk) == 0:
                label = 0
            else:
                c = np.unique(y)
                for ci in c:
                    check = np.array(y == ci)
                    count = np.sum(rk * check)
                    vote = np.append(vote, [count])
                label = np.argmax(vote)
            centroid_labels = np.append(centroid_labels, label)

        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        d2 = np.empty((0, N), float)
        for i in range(self.n_cluster):
            vt = self.centroids[i] - x
            l2 = np.sum(vt * vt, axis=1)
            d2 = np.append(d2, [l2], axis=0)
        r = np.argmin(d2, axis=0)
        labels = np.take(self.centroid_labels, r)
        return labels


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################

    # L1 distance
    def distance(x1, x2):
        return np.sqrt(np.sum(np.square(x1 - x2)))

    # find min dist point for one point and selected centers
    def nearest(point, cluster_centers):
        min_dist = np.inf
        m = np.shape(cluster_centers)[0]
        ind = 0
        for i in range(m):
            d = distance(point, cluster_centers[i])
            if min_dist > d:
                min_dist = d
                ind = i
        return code_vectors[ind]

    M, N, D = image.shape
    newimage = image.copy()
    for i in range(M):
        for j in range(N):
            newimage[i][j] = nearest(image[i][j], code_vectors)
    return newimage

