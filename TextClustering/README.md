# Text Clustering

The purpose of the assignment is to implement DBSCAN Algorithm for the Input data consists of 8580 text records in sparse format. DBSCAN Code is uploaded in .py format and lsh.py file for indexing.

## Approach:
1. Used Class Activity code to convert text records in sparse format to CSR Matrix.
2. Used Inverse Document frequency to scale matrix and normalize its rows
3. Used DR - TruncatedSVD and reduced the components to 2 (higher components did not produce better results)
4. Used sklearn library to precompute pairwise distance matrix using cosine similarity.
5. Implemented the following DBSCAN Pseudocode (took inspiration from DBSCAN wiki and Class slides)

### Find Nearest Neighbours using LSH
findK(data, lsh, index):
    lastelement = len(data) - 1
    if index == lastelement:
        findneighours = lsh.findNeighbors(data[index:], k=len(Distancematrix DB))
    else:
        findneighours = lsh.findNeighbors(data[index:index+1], k=len(Distancematrix DB))
return findneighours

### DBSCAN Algorithm
DBSCAN(Distancematrix DB, lsh, eps, minPts) {
    labels = [-1] * len(data)
    currentcluster = 0                                                                  /* Cluster counter */
    for each index in dataset DB {
        if not label(index)  -1 then continue
            Neighbors  = selectRegion(DB, lsh, index, eps, labels)      /* Find neighbors */
            if len(Neighbors) < minPts then {                                       /* Density check */
                label(index) = 0
                border.append(index)                                                      /* Label as Border and later assign those points to one of the cluster */
            }
            else
                currentcluster = currentcluster + 1                                             /* next cluster label */
                createCluster(Distancematrix DB, lsh, labels, index, neighborpoints, currentcluster, eps, minpts) /* find core points */
return labels

### Function to create the Cluster for core points
createCluster(Distancematrix DB, lsh, labels, index, neighborpoints, currentcluster, eps, minpts)
    labels[index] = currentcluster
    i = 0
    while i < len(neighborpoints):     /* expand  the cluster
    Np = neighborpoints[i]
    if labels[Np] == -1:                 /* if its noise - assign the clusterID to that index
        labels[Np] = currentcluster
    elif labels[Np] == 0         /* if its been labelled  as border - assign the clusterID to that index and look for its neighbors
        labels[Np] = currentcluster
        Npoints = selectRegion(data, lsh, Np, eps, labels)
        newlist = [x for x in Npoints if x not in neighborpoints]    /* Add new neighbors to neighbourslist only if its not there*/
        if len(newlist) >= minpts:
            neighborpoints = neighborpoints + newlist
        i += 1

### Function to check if the nearest neighbors are below eps value
selectRegion(Distancematrix DB, lsh, index, eps, labels)
    neighbors = []                                           /* Label initial point */
    kpoints = findK(Distancematrix DB, lsh, index)                     [I used lsh to get nearest neighbors)
    for i, value in np.ndenumerate(kpoints) {                             /* check if the nearest neighbours are already been clustered */
        if euclidean(index, value) ≤ eps then {                      /* I used euclidean distance and check epsilon */
            neighbors.append(value)                          /* Add to result */
}
return neighbors

6. When I used euclidean distance initially for DBSCAN to calculate the distance for the chosen point and all the other points,  my implementation took a while to compute

#### REASON WHY I CHOSE TO USE LSH
I addressed the above problem on stack exchange https://datascience.stackexchange.com/questions/30581/what-is-slowing-down-classic-dbscan-algorithm and a stack exchange user suggested me to use index structures for DBSCAN to be fast. For indexing, I chose to use LSH.
I used clsh (Professor’s code) for the distance matrix and it helped run my program fast.
Once clusters are formed and I used the below program to assign Border points to its nearest neighbour.

### Function to assign border points to its nearest neighbour
def checkclusterforborderpoints():
for j, bor in np.ndenumerate(border):
    neighbours = findK(dist, cls, bor)
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

Used Silhouette_score for internal evaluation metrics. I noticed higher the epsilon, higher the NMI score. So I increased eps value to increase NMI (because of euclidean distance?) - The above approach might not be the best one for this dataset.
