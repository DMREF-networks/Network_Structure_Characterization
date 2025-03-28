import sys
import numpy as np
import scipy as sp
from numba import njit

def URL(a, Z2): #a is the deformation parameter; Z2 is a flattened Z2 lattice
    for i in range(Z2.shape[0]):
        shift = (np.random.rand(2) - 0.5) * a
        Z2[i] += shift

def tile(flatlat,N):
    tiled = np.empty([(flatlat.shape[0])*9,2])
    im = 0
    for i in [0,-1,1]:
        for j in [0,-1,1]:
            tiled[im * flatlat.shape[0]:(im * flatlat.shape[0])+flatlat.shape[0]] = flatlat + np.array([i*N, j*N])
            im += 1
    return(tiled)

def wrap(N,flatlat):
    for i in range(flatlat.shape[0]):
        if flatlat[i,0] < 0: flatlat[i,0] += N
        if flatlat[i,0] > N: flatlat[i,0] -= N
        if flatlat[i,1] < 0: flatlat[i,1] += N
        if flatlat[i,1] > N: flatlat[i,1] -= N

@njit
def mindist(ep1, ep2, wind):
    normsq = np.sum((ep1-ep2)**2)
    t = max(0, min(1, np.dot(wind - ep1, ep2-ep1)/normsq))
    proj = ep1 + t * (ep2 - ep1)

    return(np.sum((wind-proj)**2)**0.5)

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=False, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

def eclen(cent, rad, v1, v1len, v2, v2len, edgelen):
    if v1len < rad and v2len < rad: return(edgelen)
    else:
        #print(cent, v1, v2)
        res = np.array(circle_line_segment_intersection(cent, rad, v1, v2))
        #print("Intersection points = ", res, len(res))
        if len(res) == 2: return(np.linalg.norm(res[0]-res[1]))
        elif v1len < rad: return(np.linalg.norm(res-v1))
        else: return(np.linalg.norm(res-v2))

def gabriel_graph_2d(points):
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


def remove_duplicate_rows(arr):
    dtype = [('min', arr.dtype), ('max', arr.dtype)]
    structured = np.empty(arr.shape[0], dtype=dtype)
    structured['min'] = np.minimum(arr[:, 0], arr[:, 1])
    structured['max'] = np.maximum(arr[:, 0], arr[:, 1])
    _, unique_indices = np.unique(structured, return_index=True)
    unique_indices.sort()
    return arr[unique_indices]

def restructure_array_numpy(triangles):
    reshaped = triangles
    output = np.empty((len(reshaped) * 3, 2), dtype=triangles.dtype)
    output[::3, 0] = reshaped[:, 0]
    output[::3, 1] = reshaped[:, 1]
    output[1::3, 0] = reshaped[:, 1]
    output[1::3, 1] = reshaped[:, 2]
    output[2::3, 0] = reshaped[:, 2]
    output[2::3, 1] = reshaped[:, 0]
    return output

def delaunay_graph_2d(points):
    # Perform Delaunay triangulation
    delaunay = sp.spatial.Delaunay(points)
    triangles = np.array(delaunay.simplices)

    # Convert triangles to edges
    edges = restructure_array_numpy(triangles)

    # Remove duplicate edges
    unique_edges = remove_duplicate_rows(edges)

    return unique_edges

def Z2_gen(N): #N is the number of sites along one lattice vector
    hold = np.zeros([N,N,2])
    for i in range(N):
        hold[i,:,0] = np.linspace(0,N-1,N)
    for j in range(N):
        hold[:,j,1] = np.linspace(0,N-1,N)
    return(hold)

#Square root of the number of particles in the system
N = int(sys.argv[1])
#Parameter describing the degree of local translational disorder in the URL point patterns
a = float(sys.argv[2])
#Specific identifying name for particular simulation
simtag = str(sys.argv[3])
#Configuration type
ctype = str(sys.argv[4])
#Tessellation type (C,V,G,D)
ntype = str(sys.argv[5])

#If loading a configuration, the path to the file is loaded here
##Will read in a .txt file whose lines are the global coordinates of each point in the configuration
fn = 0
if ctype == "load":
    fn = str(sys.argv[6])

#Generating/loading point pattern
if ctype == 'poi': flatlat = np.random.rand(N**2,2)*N
elif ctype == 'load':  
    flatlat = np.zeros([N**2,2])
    with open(fn, 'r') as f:
        for i in range(N**2):
            flatlat[i] = f.readline().split(' ')

else: 
    ctype = 'URL'
    lat = Z2_gen(N)
    flatlat = np.reshape(lat, [N**2,2])
    URL(a, flatlat)         

if ctype == 'poi': np.savetxt("./ELV_poi_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_config.txt", flatlat)
elif ctype == 'URL': np.savetxt("./ELV_URL_a"+str(a)+"_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_config.txt", flatlat)

#Making sure every point is inside the simulation box and making the nearest neighbor images
wrap(N, flatlat)
tURL = tile(flatlat,N)

#Generating list of edges from the choice of spatial tessellation
if ntype == 'D':
    using = np.array(delaunay_graph_2d(tURL)[:,:2], dtype=int)
elif ntype == 'G':
    using = np.array(gabriel_graph_2d(tURL)[:,:2], dtype=int)
elif ntype == 'V':
    vor = sp.spatial.Voronoi(tURL)
    using = vor.ridge_vertices
    using = remove_duplicate_rows(np.array(using))
    using = using[np.argwhere(np.sum(using>=0,axis=1)==2).flatten()]
    tURL = vor.vertices
elif ntype == 'C':
    tri = sp.spatial.Delaunay(tURL)
    newpoints = np.zeros([tri.simplices.shape[0],2])
    for i in range(newpoints.shape[0]):
        newpoints[i] = np.mean(tURL[tri.simplices[i]], axis = 0)
    tURL = newpoints
    using = []
    for i in range(tri.simplices.shape[0]):
        for j in range(3):
            if tri.neighbors[i][j] != -1: using.append([i,tri.neighbors[i][j]])
    using = remove_duplicate_rows(np.array(using))
    
using = np.array(using)

###Settings to change the number of windows used in the calculation or resolution in window radius
Nwind = 250
reso = 500

wind = np.random.rand(Nwind,2)*N
Rs = np.logspace(-2,np.log10(N/4),reso)
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
