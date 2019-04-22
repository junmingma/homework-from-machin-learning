import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception(
            #    'Implement initialization of variances, means, pi_k using k-means')
            self.means, R, count = KMeans(self.n_cluster, self.max_iter, self.e).fit(x)
            r = np.eye(self.n_cluster)[R]
            N_k = np.sum(r, axis=0)
            self.pi_k = np.sum(r, axis = 0) / N
            self.variances = np.zeros((self.n_cluster,D,D))
            for k in range(self.n_cluster):
                x_k = x[R==k]
                diff = x_k-self.means[k]  
                self.variances[k] = np.dot(diff.T, diff)/N_k[k]

            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception(
            #    'Implement initialization of variances, means, pi_k randomly')
            self.means = np.random.rand(self.n_cluster,D)
            self.variances = np.full((self.n_cluster,D,D), np.identity(D))
            self.pi_k = np.full(self.n_cluster, 1/self.n_cluster)
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement fit function (filename: gmm.py)')

        i_ter = 0
        log_likelihood = -np.inf
        r = np.zeros((N, self.n_cluster))
        for i in range(self.max_iter):
            #E-step: compute responsibilities
            i_ter = i
            for k in range(self.n_cluster):
                while np.linalg.det(self.variances[k]) == 0:
                    self.variances[k] += 0.001 * np.identity(D)
                comp = self.pi_k[k]/np.sqrt((2*np.pi)**D*np.linalg.det(self.variances[k]))
                exp = np.exp(-0.5*np.sum(np.multiply(x-self.means[k], np.dot(x-self.means[k], np.linalg.inv(self.variances[k]))), axis=1))
                r[:,k] = comp*exp
            log_likelihood_new = np.sum(np.log(np.sum(r, axis=1)))
            r = (r.T / np.sum(r, axis=1)).T
            Nk = np.sum(r, axis=0)

            #Mstep
            self.pi_k = Nk / N

            for k in range(self.n_cluster):
                self.means[k] = np.transpose(np.sum(r[:, k] * np.transpose(x), axis=1)) / Nk[k]
                self.variances[k] = np.dot(np.multiply(np.transpose(x - self.means[k]), r[:, k]), x - self.means[k]) / Nk[k]

            # Compare l and l_new
            if np.abs(log_likelihood - log_likelihood_new) <= self.e:
                break
            log_likelihood = log_likelihood_new
        return i_ter
        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement sample function in gmm.py')
        # generate N samples scattered in different K clusters according to pi_k

        D = self.means.shape[1]
        samples = np.zeros((N,self.means.shape[1]))
        k = np.random.choice(self.n_cluster, N, p=self.pi_k)
        for i in range(k.shape[0]):
            mean = self.means[k[i]]
            var = self.variances[k[i]]
            samples[i] = np.random.multivariate_normal(mean, var)

        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement compute_log_likelihood function in gmm.py')
        N, D = x.shape
        r = np.zeros((N,self.n_cluster))
        for k in range(self.n_cluster):
            while np.linalg.det(self.variances[k]) == 0:
                self.variances[k] += 0.001 * np.identity(D)
            comp = pi_k[k]/np.sqrt((2*np.pi)**D*np.linalg.det(self.variances[k]))
            exp = np.exp(-0.5*np.sum(np.multiply(x-means[k], np.dot(x-means[k], np.linalg.inv(self.variances[k]))), axis=1))
            r[:,k] = comp*exp
        log_likelihood = float(np.sum(np.log(np.sum(r, axis=1))))
        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception('Impliment Guassian_pdf __init__')
            D = len(variance)
            while np.linalg.det(self.variance) == 0:
                self.variance = self.variance + 0.001 * np.identity(D)
            self.inv = np.linalg.inv(self.variance)
            self.c = ((2  *np.pi) **D) * np.linalg.det(self.variance)
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception('Impliment Guassian_pdf getLikelihood')
            x_min_mean = x-self.mean
            product = np.dot(np.dot(x_min_mean,self.inv),x_min_mean.reshape(-1,1))
            p = np.exp((-0.5) * product) / np.sqrt(self.c)
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
