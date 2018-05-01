
import matplotlib as plt
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
import numpy as np
import random
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import silhouette_score
from numpy import square, sqrt
from scipy.spatial.distance import cosine
from lsh import clsh, jlsh, generateSamples, findNeighborsBrute, recall


def csr_build(dataIndex, value):
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0
    n = 0

    for (d,v) in zip(dataIndex, value):
        l = len(d)
        for j in range(l):
            ind[int(j) + n] = d[j]
            val[int(j) + n] = v[j]

        ptr[i+1] = ptr[i] + l
        n += l
        i += 1

    mat = csr_matrix((val, ind, ptr), shape=(nrows, max(ind)+1), dtype=np.double)
    mat.sort_indices()

    return mat

def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = float(1.0/np.sqrt(rsum))
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat


data = open('/Users/whiplash/SJSU/Semester 2/CMPE 255/Assignments/Program 3/train.txt', 'r')
documents = []
for i in data:
    documents.append(i.rstrip().split(" "))
index = []
value = []
for d in documents:
    d_index = []
    d_value = []
    for i in range(0,len(d),2):      
        d_index.append(d[i])
    for j in range(1,len(d),2):     
        d_value.append(d[j])
    dataIndex.append(d_index)
    value.append(d_value)
nrows = len(documents)
idx = {}
tid = 0
nnz = 0
ncol = 0
max = []
for i in index:
    nnz += len(i)
    _max.append(max(i))
    for words in d:
        if words not in idx:
            idx[w] = tid
            tid += 1

matrix = csr_build(dataIndex, value)
csrIdf = csr_idf(matrix, copy=True)
csrNormalized = csr_l2normalize(csrIdf, copy=True)
csr = csrNormalized.toarray()
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2).fit_transform(csr)

svd


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
dist = pairwise_distances(svd, metric='cosine')


# In[16]:


print(dist)


# In[17]:


print(dist[0])


# In[20]:

cls = clsh(dist)


# In[285]:
## DBSCAN Algorithm

border = []
#  Return labels of the cluster. labels start from 1, -1 is noise intially and assign to k+1 label later
def dbscanalgo(data, lsh, eps, minpts):
    labels = [-1] * len(data)
    currentcluster = 0
#     visited = []
    for index in range(len(data)):
        print(index)
        if not (labels[index] == -1):
            continue
        neighborpoints = selectRegion(data, lsh, index, eps, labels)
        if len(neighborpoints) < minpts:
            labels[index] = 0 #setting border points as 0
            border.append(index)
        else:
            print(Counter(labels))
            currentcluster += 1
            print("currentcluster", currentcluster)
            createCluster(data, lsh, labels, index, neighborpoints, currentcluster, eps, minpts)
    print("Done")
    return labels

##Function to create the Cluster for core points
def createCluster(data, lsh, labels, index, neighborpoints, currentcluster, eps, minpts):
    labels[index] = currentcluster
    i = 0
    while i < len(neighborpoints):    
        Np = neighborpoints[i]
        if labels[Np] == -1:
            labels[Np] = currentcluster    
        elif labels[Np] == 0:
            labels[Np] = currentcluster            
            Npoints = selectRegion(data, lsh, Np, eps, labels) 
            newlist = [x for x in Npoints if x not in neighborpoints]
            if len(newlist) >= minpts:
                neighborpoints = neighborpoints + newlist          
        i += 1     
    
##Function to check if the nearest neighbors are below eps value
def selectRegion(data, lsh, index, eps, labels):
    neighbors = []
    kpoints = findKneighbours(data, lsh, index)
    for i, value in np.ndenumerate(kpoints):
        if value != -1:
            if labels[value] == 0 or labels[value] == -1: 
            #neighbour either should be a noise or unclassified 
                distance = np.linalg.norm(data[index] - data[value]) # calculate euclidean distance between two points
                if distance <= eps: #more towards 1, more similar
                    neighbors.append(value)
    return neighbors

# def ktree(data, index, minpts):
#     Neighbors = kdtree.query_ball_point(data[index,:],minpts)
##Find Nearest Neighbours using LSH
def findKneighbours(data, lsh, index):
    lelement = len(data) - 1
    if index == lastelement:
        findneighours = lsh.findNeighbors(data[index:], k=len(data))
    else:
        findneighours = lsh.findNeighbors(data[index:index+1], k=len(data))
    return findneighours


# In[286]:


clusterL = dbscanalgo(dist, cls, 15, 21)


# In[287]:


Counter(clusterL)


# In[288]:


print(clusterL)


# In[289]:


print(len(clusterL))


# In[290]:


print(len(border))


# In[291]:


print(border)


# In[265]:


len(border)


# In[266]:


borderlist = []


# In[267]:


def checkclusterforborderpoints():
    for j, bor in np.ndenumerate(border):
        neighbours = findKneighbours(dist, cls, bor)
        closestneighbourdistance = 1
        closestneighbour = 0
        for i, value in np.ndenumerate(neighbours):
            if value != -1:
                distance = np.linalg.norm(dist[bor] - dist[value])
                if distance > 0:
                    if closestneighbourdistance > distance:
                        closestneighbourdistance = distance
                        closestneighbour = value
        borderlist.append(closestneighbour)

checkclusterforborderpoints()

lab = clusterL


for i, value in enumerate(border):
    lab[value] = lab[borderlist[i]]

Counter(lab)

prediction = open("prediction.txt", "w")
for i in lab:  
    prediction.write(str(i) +'\n')
prediction.close()


bestscore = 0
minbest = 0
for minpts in range(5, 25, 2):
        print("minpts", minpts)
        matrixlabels = dbscanalgo(dist, cls, 15, minpts)
        silhouette_score = silhouette_score(csr, matrixlabels)
        print("Score", silhouette_score)
        Xaxis.append(minpts)
        Yaxis.append(silhouette_score)
        if bestscore < silhouette_score:
            bestscore = silhouette_score
            minbest = minpts


plt.xlabel('Number of minpts')
plt.ylabel('Silhoutte Score')
plt.plot(Xaxis, Yaxis)


plt.show()

plt.savefig('graph.png')