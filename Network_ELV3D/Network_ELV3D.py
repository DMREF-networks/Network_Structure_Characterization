import sys
import numpy as np
import scipy as sp
from numba import njit

def restructure_array_numpy(reshaped):
    output = np.empty((len(reshaped) * 6, 2), dtype=reshaped.dtype)
    output[::6, 0] = reshaped[:, 0]
    output[::6, 1] = reshaped[:, 1]
    output[1::6, 0] = reshaped[:, 1]
    output[1::6, 1] = reshaped[:, 2]
    output[2::6, 0] = reshaped[:, 2]
    output[2::6, 1] = reshaped[:, 3]
    output[3::6, 0] = reshaped[:, 3]
    output[3::6, 1] = reshaped[:, 0]
    output[4::6, 0] = reshaped[:, 0]
    output[4::6, 1] = reshaped[:, 2]
    output[5::6, 0] = reshaped[:, 1]
    output[5::6, 1] = reshaped[:, 3]
    return output


def remove_duplicate_rows(arr):
    dtype = [('min', arr.dtype), ('max', arr.dtype)]
    structured = np.empty(arr.shape[0], dtype=dtype)
    structured['min'] = np.minimum(arr[:, 0], arr[:, 1])
    structured['max'] = np.maximum(arr[:, 0], arr[:, 1])
    _, unique_indices = np.unique(structured, return_index=True)
    unique_indices.sort()
    return arr[unique_indices]

def eclen(cent, rad, v1, v1len, v2, v2len, edgelen):
    if v1len < rad and v2len < rad: return(edgelen)
    else:
        res = np.array(line_sphere_intersection(cent, rad, v1, v2))
        if len(res) == 2: return(np.linalg.norm(res[0]-res[1]))
        elif v1len < rad: return(np.linalg.norm(res-v1))
        else: return(np.linalg.norm(res-v2))        

def line_sphere_intersection(c,r,p1, p2, full_line=False, tangent_tol=1e-9):
    """
    Finds the intersection points between a line segment and a sphere.

    Args:
        p1 (numpy.array): The starting point of the line segment.
        p2 (numpy.array): The ending point of the line segment.
        c (numpy.array): The center of the sphere.
        r (float): The radius of the sphere.

    Returns:
        list: A list of intersection points. An empty list if there are no intersections.
    """

    # Line direction vector
    d = p2 - p1

    # Coefficients of the quadratic equation
    a = np.dot(d, d)
    b = 2 * np.dot(d, p1 - c)
    c = np.dot(p1 - c, p1 - c) - r**2

    # Discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return []
    elif discriminant == 0:
        # One intersection point
        t = -b / (2 * a)
        if 0 <= t <= 1:
            return [p1 + t * d]
        else:
            return []
    else:
        # Two intersection points
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

        intersection_points = []
        if 0 <= t1 <= 1:
            intersection_points.append(p1 + t1 * d)
        if 0 <= t2 <= 1:
            intersection_points.append(p1 + t2 * d)

        return intersection_points


@njit
def mindist(ep1, ep2, wind):
    normsq = np.sum((ep1-ep2)**2)
    t = max(0, min(1, np.dot(wind - ep1, ep2-ep1)/normsq))
    proj = ep1 + t * (ep2 - ep1)

    return(np.sum((wind-proj)**2)**0.5)

def delaunay_graph_3d(points):
    # Perform Delaunay triangulation
    delaunay = sp.spatial.Delaunay(points)
    triangles = np.array(delaunay.simplices)

    # Convert triangles to edges
    edges = restructure_array_numpy(triangles)

    # Remove duplicate edges
    unique_edges = remove_duplicate_rows(edges)

    return unique_edges

def gabriel_graph_3d(points):
    """
    Compute the Gabriel graph of a set of points in 2D or 3D space.

    Parameters:
    - points: (N, D) numpy array where N is the number of points and D is the dimension (2 or 3).

    Returns:
    - edge_array: (M, 3) numpy array where each row represents an edge in the format [node1, node2, weight].
    """
    # Build a KD-tree for efficient neighbor searches
    tree = sp.spatial.cKDTree(points)
    # Compute the Delaunay triangulation
    tri = sp.spatial.Delaunay(points)
    simplices = tri.simplices  # Indices of points forming the simplices

    # Generate all possible edges from the simplices
    if simplices.shape[1] == 3:  # 2D case
        edges = np.vstack([simplices[:, [0, 1]],
                           simplices[:, [1, 2]],
                           simplices[:, [2, 0]]])
    elif simplices.shape[1] == 4:  # 3D case
        edges = np.vstack([simplices[:, [0, 1]],
                           simplices[:, [0, 2]],
                           simplices[:, [0, 3]],
                           simplices[:, [1, 2]],
                           simplices[:, [1, 3]],
                           simplices[:, [2, 3]]])
    else:
        raise ValueError('Input points must be 2D or 3D.')

    # Sort and remove duplicate edges
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    # Compute midpoints and radii for the Gabriel condition
    i = edges[:, 0]
    j = edges[:, 1]
    points_i = points[i]
    points_j = points[j]
    midpoints = (points_i + points_j) / 2
    radii = 0.5 * np.linalg.norm(points_i - points_j, axis=1)
    weights = 2 * radii  # Edge weights are the Euclidean distances

    # Query KD-tree to find neighboring points within the radius
    idx_list = tree.query_ball_point(midpoints, radii)

    # Build the edge list for the Gabriel graph
    edge_list = []
    for k in range(len(edges)):
        idx = set(idx_list[k]) - {i[k], j[k]}
        if not idx:
            edge_list.append([i[k], j[k], weights[k]])

    edge_array = np.array(edge_list)
    return edge_array


def Z3_gen(N): #N is the number of sites along one lattice vector
    x, y, z = np.mgrid[0:N, 0:N, 0:N]
    coords = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    return np.array(coords, dtype='float')
    
def URL_3D(a, Z3): #a is the deformation parameter; Z2 is a flattened Z2 lattice
    for i in range(Z3.shape[0]):
        shift = (np.random.rand(3) - 0.5) * a
        Z3[i] += shift

def wrap_3D(N,flatlat):
    for i in range(flatlat.shape[0]):
        if flatlat[i,0] < 0: flatlat[i,0] += N
        if flatlat[i,0] > N: flatlat[i,0] -= N
        if flatlat[i,1] < 0: flatlat[i,1] += N
        if flatlat[i,1] > N: flatlat[i,1] -= N
        if flatlat[i,2] < 0: flatlat[i,2] += N
        if flatlat[i,2] > N: flatlat[i,2] -= N

def tile_3D(flatlat,N):
    tiled = np.empty([(flatlat.shape[0])*27,3])
    im = 0
    for i in [0,-1,1]:
        for j in [0,-1,1]:
            for k in [0,-1,1]:
                tiled[im * flatlat.shape[0]:(im * flatlat.shape[0])+flatlat.shape[0]] = flatlat + np.array([i*N, j*N, k*N])
                im += 1
    return(tiled)

#Cube root of the number of particles in the system
N = int(sys.argv[1])
#Parameter describing the degree of local translational disorder in the URL point patterns
a = float(sys.argv[2])
#Specific identifying name for particular simulation
simtag = str(sys.argv[3])
#Configuration type
ctype = str(sys.argv[4])
#Tessellation type (C,V,G,D)
ntype = str(sys.argv[5])

#If loading a configuration, the path to the file is loaded here##Will read in a .txt file whose lines are the global coordinates of each point in the configuration
fn = 0
if ctype == "load":
    fn = str(sys.argv[6])

#Generating/loading the point pattern
if ctype == 'poi': flatlat = np.random.rand(N**3,3)*N
elif ctype == 'load':  
    flatlat = np.zeros([N**3,3])
    with open(fn, 'r') as f:
        for i in range(N**3):
            flatlat[i] = f.readline().split(' ')
else: 
    ctype = 'URL'
    flatlat = Z3_gen(N)
    URL_3D(a, flatlat)

if ctype == 'poi': np.savetxt("./ELV_poi_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_config.txt", flatlat)
elif ctype == 'URL': np.savetxt("./ELV_URL_a"+str(a)+"_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_config.txt", flatlat)

#Making sure each each point is inside of the fundamental cell and generating all 26 of the nearest images
wrap_3D(N, flatlat)
tURL = tile_3D(flatlat,N)

###Remove all points more than ~L/3.85 outside the fundamental cell to speed up calculations
cURL = (tURL)/N
td = np.unique(np.append(np.argwhere(cURL>1.3)[:,0], np.argwhere(cURL<-0.3)[:,0]))

tURL = np.delete(tURL,td,axis=0)

#Generating the set of edges from the particular choice of spatial tessellation
if ntype == 'D':
    using = np.array(delaunay_graph_3d(tURL)[:,:2], dtype=int)
elif ntype == 'G':
    using = np.array(gabriel_graph_3d(tURL)[:,:2], dtype=int)
elif ntype == 'V':
    vor = sp.spatial.Voronoi(tURL)
    using = []
    for i in vor.ridge_vertices:
        if np.sum(np.array(i)<0) == 0:
            using.append([i[0],i[-1]])
            for j in range(len(i)-1):
                using.append([i[j], i[j+1]])
    using = np.sort(np.array(using),axis=1)
    using = remove_duplicate_rows(using)
    tURL = vor.vertices
elif ntype == 'C':
    tri = sp.spatial.Delaunay(tURL)
    newpoints = np.zeros([tri.simplices.shape[0],3])
    for i in range(newpoints.shape[0]):
        newpoints[i] = np.mean(tURL[tri.simplices[i]], axis = 0)
    tURL = newpoints
    using = []
    for i in range(tri.simplices.shape[0]):
        for j in range(4):
            if tri.neighbors[i][j] != -1: using.append([i,tri.neighbors[i][j]])
    using = remove_duplicate_rows(np.array(using))

using = np.array(using)

#Setting to change the number of windows used in the calculation or resolution in window radius
Nwind = 500
reso = 500

wind = np.random.rand(Nwind,3)*N
Rs = np.logspace(-4,np.log10(N/4),reso)
uselen = np.zeros(using.shape[0])
for i in range(using.shape[0]):
    uselen[i] = np.linalg.norm(tURL[using[i,0]]-tURL[using[i,1]])

tovar = np.zeros([Nwind, reso])

for i in range(Nwind):
    distto = np.zeros(using.shape[0])
    verttowind = np.zeros(tURL.shape[0])
    for j in range(using.shape[0]):
        distto[j] = mindist(tURL[using[j,0]], tURL[using[j,1]], wind[i])
    for j in range(verttowind.shape[0]):
        verttowind[j] = np.linalg.norm(wind[i]-tURL[j])
    for j in range(Rs.shape[0]):
        etc = np.argwhere(distto < Rs[j]).flatten()
        for k in etc:

            tr = eclen(wind[i], Rs[j], tURL[using[k,0]], verttowind[using[k,0]], tURL[using[k,1]],  verttowind[using[k,1]], uselen[k])
            tovar[i,j] += (tr)


if ctype == 'load': 
    np.save("./ELV_ld_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag), tovar)
    np.save("./ELV_ld_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_Rs", Rs)
elif ctype == 'poi': 
    np.save("./ELV_poi_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag), tovar)
    np.save("./ELV_poi_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_Rs", Rs)
elif ctype == 'URL': 
    np.save("./ELV_URL_a"+str(a)+"_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag), tovar)
    np.save("./ELV_URL_a"+str(a)+"_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_Rs", Rs)
