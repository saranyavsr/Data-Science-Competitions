#!/usr/bin/python
r""" Example of Approximate Nearest Neighbors using Locality Sensitive Hashing (LSH)
	with Hamming Distance, Jaccard Coefficient and Cosine Similarity.
	Given a training/database set of objects and a test/query set of objects,
	find the k-Nearest Neighbors (k-NN) for each of the objects in the query set
	and report the accuracy of the result (measured as average recall, i.e., the mean
	of the ratios, number of true neighbors found / number of neighbors sought).
	
	Note: This is an academic implementation meant to understand the general LSH
	search mechanism. Much better implementations and extensions of the basic LSH 
	algorithm exist in practice.
"""
import time
import argparse
import numpy as np
from random import randint
from numpy.linalg import norm
from operator import itemgetter
from collections import defaultdict
from sklearn.datasets.samples_generator import make_blobs


__all__ = ['blsh', 'clsh', 'hlsh', 'jlsh']

class blsh(object):
    r""" Example data structure for computing Approximate Nearest Neighbors 
    using Locality Sensitive Hashing (LSH) with Hamming Distance, 
    Jaccard Coefficient and Cosine Similarity.
    
    This is a base abstract class that is extended by three concrete classes:
    - clsh implements an LSH table for Cosine similarity. 
    - hlsh implements an LSH table for Hamming distance. 
    - jlsh implements an LSH table for Jaccard coefficient. 
    
    Example usage:
        L = clsh(X, ntables=10, nfunctions=10)
        N = L.findNeighbors(Y, k=5)
            
    Attributes
    ----------
    ntables : int
        Number of hash tables to create
    nfunctions : int
        Number of functions to use for each table
        
    Notes:
    ------
    
    This is an academic implementation meant to understand the general LSH
    search mechanism. Much better implementations and extensions of the basic LSH 
    algorithm exist in practice.
    """
    
    def __init__(self, X, ntables=10, nfunctions=10):
        r""" Initialize an elsh structure.
        
        Parameters:
            X: numpy 2D array (matrix). Each object takes up a row of X.
            ntables: int, Number of hash tables to create.
            nfunctions: int, Number of functions to use for each table.
        """
        self.X = np.copy(X)
        self.ntables = ntables
        self.nfunctions = nfunctions
        self.nsims = 0 # counter for number of computed similarities
        self.sim = True # whether this is a similarity function, not a distance function
        # hash functions, one set of nfunctions functions for each table
        self.hfts = [{} for _ in range(ntables)]    
        # the hash tables
        self.tables = [defaultdict(set) for _ in range(ntables)]
        # hash data in X
        self._hashData(self.X)

    def hash(self, x, tid=0, fid=0):
        r""" Hash vector x using the function fid in table tid.
        Implemented by subclasses.
        """
        raise NotImplementedError()

    def signature(self, x, tid=0):
        r""" Get the signature (bucket ID) of x in table tid.
        
        Return: string
        """
        hk = ''
        for fid in range(self.nfunctions):
            h = self.hash(x, tid, fid)
            # We use a simple table hash key, concatenating the string 
            # representations of the hash function values returned in h.
            # This is equivalent to the AND operation, i.e., two objects
            # must have the same h value for all nfunctions functions to
            # be added to the same bucket (think set intersection!).
            hk += '%d' % h
        return hk
        
    @staticmethod
    def prox(x, y):
        r""" Compute the proximity (similarity or distance) of two vectors
        """
        raise NotImplementedError()
    
    def _hashData(self, X):
        r""" Hash a set of data points.
        We take an AND + OR approach to constructing hashes. We concatenate 
        hash functions in each table to decrease the probability of the collision
        of distant points as much as possible. This, in turn, increases precision.
        By considering points found in multiple tables, we increase the probability
        of the collision of close points, which increases recall. 
        """
    
        for i in range(X.shape[0]):
            x = X[i,:] # row i in X
            for tid in range(self.ntables):
                table = self.tables[tid]
                hk = self.signature(x, tid)
                table[hk].add(i) # add row i to bucket hk in table tid

    def search(self, y, k=5):
        r""" Find neighbors for object y given an already constructed set of LSH tables,
        returning the k nearest neighbors found.
        We hash y in each table, and compare against the union of all found object ids.
        
        Return: a list of tuples of neighbor id and similarity value
        """
        cands = set()
        for tid in range(self.ntables):
            table = self.tables[tid]
            hk = self.signature(y, tid)
            cands = cands.union(table[hk])
        sims = [(i, self.prox(self.X[i,:], y)) for i in cands]
        self.nsims += len(sims)
        sims.sort(key=itemgetter(1), reverse=self.sim) # sort by value
        if len(sims) < k:
            k = len(sims)
            
        return sims[:k]
    
    def findNeighbors(self, Y, k=5, retvals=False):
        r""" Find k-NN using for a set of objects using the LSH structure
        
        Return: An array containing up to k neighbors for each row in Y. If less than
        k neighbors are found, the remaining IDs will be -1. If retval is True, a second
        array will be returned with corresponding values for the nearest neighbors.
        """
        nte = Y.shape[0]
        nbrs = np.full((nte, k), -1, dtype=np.int) # in case some searches return less than k items
        if retvals is True:
            nvals = np.zeros((nte, k), dtype=np.double)
	# reset nsims
        self.nsims = 0
        
        for i in range(nte):
            sims = self.search(Y[i,:], k=k)
            nk = len(sims)
            if not nk:
                continue
            nbr, sim = zip(*sims) # split neighbor ids and values
            nbrs[i, :nk] = nbr
            if retvals:
                nvals[i, :nk] = sim
        
        if retvals:
            return nbrs, nvals
        return nbrs
    

class clsh(blsh):
    r""" LSH table for Cosine similarity
    """
    
    def hash(self, x, tid=0, fid=0):
        r""" Hash vector x using the function fid in table tid.
        
        Return: binary value (1|0)
        """
        assert tid < self.ntables
        assert fid < self.nfunctions
        
        hft = self.hfts[tid] # a map containing hash functions
        m = len(x) # number of features
        
        if fid not in hft:
            # generate random unit vector
            r = np.random.randn(m)
            r = r/norm(r)
            hft[fid] = r
        
        r = hft[fid]     # random unit vector
        hr = x.dot(r.T)  # dot product of x with r
        return 1 if hr > 0 else 0
        
    @staticmethod
    def prox(x, y):
        r""" Compute the Cosine similarity of two vectors
        
        Return: double
        """
        return x.dot(y.T)/(norm(x)*norm(y))
    

class jlsh(blsh):
    r""" LSH table for Jaccard coefficient
    """
    
    def hash(self, x, tid=0, fid=0):
        r""" Hash vector x using the function fid in table tid.
        
        Return: int
        """
        assert tid < self.ntables
        assert fid < self.nfunctions
        
        hft = self.hfts[tid] # a map containing hash functions
        m = len(x) # number of features
        
        if fid not in hft:
            # generate random permutation of feature ids
            hft[fid] = np.random.permutation(m)
            
        p = hft[fid]
        for i in range(m):
            if x[ p[i] ]:
                return p[i] # first feature with non-zero value in x according to permutation p
        return m
        
    @staticmethod
    def prox(x, y):
        r""" Compute the Jaccard coefficient of two vectors
        
        Return: double
        """
        z = x+y
        return np.sum(z == 2) / (0.0 + np.sum(z > 0))



class hlsh(blsh):
    r""" LSH table for Hamming distance
    """
    
    def __init__(self, X, ntables=10, nfunctions=10):
        r""" Initialize an hlsh structure.
        
        Parameters:
            X: numpy 2D array (matrix). Each object takes up a row of X.
            ntables: int, Number of hash tables to create.
            nfunctions: int, Number of functions to use for each table.
        """
        super(hlsh, self).__init__(X, ntables=ntables, nfunctions=nfunctions)
        self.sim = False
    
    def hash(self, x, tid=0, fid=0):
        r""" Hash vector x using the function fid in table tid.
        
        Return int
        """
        assert tid < self.ntables
        assert fid < self.nfunctions
        
        hft = self.hfts[tid] # a map containing hash functions
        m = len(x) # number of features
        
        if fid not in hft:
            # generate random feature id
            hft[fid] = randint(0, m-1)
            
        return x[ hft[fid] ] # value in x for the chosen random feature 
        
    @staticmethod
    def prox(x, y):
        r""" Compute the Hamming distance of two vectors
        
        Return: int
        """
        return np.sum(x!=y)


def get_args():
    r""" Parse arguments for the program    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-sim", "--sim", type=str, default="jac", help="Which similarity to compute for, e.g. cos|jac")
    parser.add_argument("-n", "--nsamples", type=int, default=1000, help="Number of random samples to generate")
    parser.add_argument("-m", "--nfeatures", type=int, default=100, help="Number of features for each sample")
    parser.add_argument("-f", "--nfunctions", type=int, default=2, help="Number of random hash functions to use")
    parser.add_argument("-t", "--ntables", type=int, default=10, help="Number of tables/signatures to create")
    parser.add_argument("-k", "--nnbrs", type=int, default=5, help="Number of nearest neighbors to find")
    
    return parser.parse_args()


def generateSamples(nsamples=10000, nfeatures=1000, nclusters=100, clusterstd=50, binary=False):
    r""" Generate random samples for the experiment
    """
    # generate samples from several clusters (blobs)
    X, _ = make_blobs(n_samples=nsamples, n_features=nfeatures, 
    	centers=nclusters, cluster_std=clusterstd)
    
    # binarize the data if necessary by taking only values above the mean
    if binary:
        mu = np.mean(X) + np.random.randn()
        X[X > mu] = 1
        X[X <= mu] = 0
        X = np.array(X, dtype=np.int)
        
    # split data into train and test sets, using a 90:10 split
    c = int(X.shape[0]/10)
    if c > 1000:
        c = 1000 # max 1000 test items
    X_te = X[:c]
    X_tr = X[c:]
    
    return X_tr, X_te


def findNeighborsBrute(X_tr, X_te, k=5, sim="cos", retvals=False):
    r""" Find k-NN using a brute-force approach
    """
    nte = int(X_te.shape[0]) # number of test samples
    ntr = int(X_tr.shape[0]) # number of training samples
    # space to store neighbors 
    nbrs = np.zeros((nte, k), dtype=np.int)
    if retvals is True:
        nvals = np.zeros((nte, k), dtype=np.double)
    
    # choose the right proximity function
    if sim == "cos":
        prox = clsh.prox
    elif sim == "ham":
        prox = hlsh.prox
    elif sim == "jac":
        prox = jlsh.prox
    else:
        raise ValueError("Incorrect proximity function. Try one of: cos, ham, or jac.")
    
    # find the nearest neighbors for each item in the test set
    for i in range(nte):
        # compute proximity of X_te[i] to all X_tr objects and sort list,
        # returning the IDs of the closest objects
        sims = [prox(X_te[i],X_tr[j]) for j in range(ntr)]
        p = np.argsort(sims)
        if sim != "ham":
            p = p[::-1] # reverse list direction
        nbrs[i,:] = p[:k]
        if retvals is True:
            nvals[i,:] = [sims[k] for k in p[:k]]
    
    if retvals is True:
        return nbrs, nvals
    return nbrs
 
def recall(nbrsTest, nbrsExact):
    r""" Compute the mean recall of the nearest neighbor search.
    For each set of neighbors, recall is defined as the number of true/exact
    neighbors returned divided by the number of requested neighbors.
    
    Return: float
    """
    acc = 0.0
    n,k = nbrsExact.shape
    for i in range(n):
        a = nbrsTest[i, :]
        b = nbrsExact[i, :]
        acc += np.intersect1d(a, b).shape[0] / float(k)
        
    return acc / float(n)

if __name__ == '__main__':
    
    args = get_args()  # command line arguments
    
    lshClasses = {"cos": clsh, "ham": hlsh, "jac": jlsh}
    
    X, Y = generateSamples(nsamples=args.nsamples, nfeatures=args.nfeatures,
        binary=False if args.sim == "cos" else True)
    
    if args.sim not in lshClasses:
        raise ValueError("Incorrect similarity choice. Try cos, ham, or jac.")
    cls = lshClasses[args.sim] # choose the correct LSH structure
    
    # find exact neighbors
    nbrsExact = findNeighborsBrute(X, Y, args.nnbrs, sim=args.sim, retvals=False)
    
    # find approximate neighbors using LSH 
    t0 = time.time()
    L = cls(X, ntables=args.ntables, nfunctions=args.nfunctions)
    nbrsTest  = L.findNeighbors(Y, k=args.nnbrs)
    T = time.time() - t0
    
    print ("LSH search time: %f s" % T)
    print ("Number of computed brute-force similarities: %d" % Y.shape[0] * X.shape[0])
    print ("Number of computed LSH similarities: %d" % L.nsims)
    print ("Accuracy: %.2f" % recall(nbrsTest, nbrsExact))


