import sys
import numpy as np
import scipy as sp
import shapely
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import imageio.v3 as iio
from numba import jit

@jit(nopython=True,fastmath=True)
def comp_mk(Kx1,Ky1,mk):
    for i in range(0,Kx1.shape[0]):
        for j in range(0,Ky1.shape[1]):
                k_x = Kx1[i][j]
                k_y = Ky1[i][j]
                mk[i][j] = np.sinc(k_x/2/np.pi)*np.sinc(k_y/2/np.pi)

def binning_procedure(Sk_norm, binset):
    binsize = np.min(Sk_norm[:,0])*binset
    bins = np.arange(binsize, np.max(Sk_norm[:,0]), binsize)
    inds = inds = np.digitize(Sk_norm[:,0], bins)
    uni_ind, count_inds = np.unique(inds, return_counts=True)
    bin2 = bins[uni_ind-1]
    Sk_b = np.zeros(len(uni_ind))
    b_b = np.zeros(len(uni_ind))

    i = 0
    j = 0
    k = 0
    while(i < Sk_norm[:,0].shape[0]):
        w = count_inds[k]
        while(j < w):
            Sk_b[k] += Sk_norm[i+j,1]
            b_b[k] += Sk_norm[i+j,0]
            j+=1
        j=0
        Sk_b[k] /= w
        b_b[k] /= w
        #Xv_final[k] /= w
        i+=w
        k+=1
    return(bin2, Sk_b)

@jit(nopython=True,fastmath=True)
def fill_vecs(Xs, kXvals, kYvals, k_grid, norms, Sk_norm):
    kv = 0
    for i in range(0,kXvals.shape[0]):
        for j in range(0,kYvals.shape[0]):
                k_grid[i][j][0] = kXvals[i]
                k_grid[i][j][1] = kYvals[j]
                norms[kv] = np.sqrt(k_grid[i][j][0]**2 + k_grid[i][j][1]**2)
                Sk_norm[kv][0] = Xs[i][j]
                Sk_norm[kv][1] = norms[kv]
                kv+=1

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

def wrap(N,flatlat): #N is the square root of the total particle number, flatlat is a 2 by N**2 matrix of particle locations
    for i in range(flatlat.shape[0]):
        if flatlat[i,0] < 0: flatlat[i,0] += N
        if flatlat[i,0] > N: flatlat[i,0] -= N
        if flatlat[i,1] < 0: flatlat[i,1] += N
        if flatlat[i,1] > N: flatlat[i,1] -= N

def Z2_gen(N): #N is the number of sites along one lattice vector
    hold = np.zeros([N,N,2])
    for i in range(N):
        hold[i,:,0] = np.linspace(0,N-1,N)
    for j in range(N):
        hold[:,j,1] = np.linspace(0,N-1,N)
    return(hold)

def URL(a, Z2): #a is the deformation parameter; Z2 is a 2 by N**2 flattened Z2 lattice
    for i in range(Z2.shape[0]):
        shift = (np.random.rand(2) - 0.5) * a
        Z2[i] += shift

def tile(flatlat,N): #Returns a supercell of the original configuration containing all nearest images in two dimensions
    tiled = np.empty([(flatlat.shape[0])*9,2])
    im = 0
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            #print(flatlat)# + np.array([i*N, j*N]))
            tiled[im * flatlat.shape[0]:(im * flatlat.shape[0])+flatlat.shape[0]] = flatlat + np.array([i*(flatlat.shape[0])**0.5, j*(flatlat.shape[0])**0.5])
            im += 1
    return(tiled)

def networkGen(N, a, beamw, namestub, fn = 0, figtype = 'URL', tess = None):
    
    #This chain of if/else statements will make one of the three configuration types used in the paper
    if figtype == 'load':
        flatlat = np.zeros([N**2,2])
        with open(fn) as f:
            for i in range(N**2):
                flatlat[i] = f.readline().split(' ')
        nameroot = "ld_N_" + str(N) + "_beamw_" + str(beamw) + "_" + namestub + "_"
    
    elif figtype == 'poi':
        flatlat = np.random.rand(N**2,2)*N
        nameroot = "POI_N_" + str(N) + "_beamw_" + str(beamw) + "_" + namestub + "_"
    
    else: 
        lat = Z2_gen(N)
        flatlat = np.reshape(lat, [N**2,2])
        URL(a, flatlat)
        wrap(N, flatlat)
        nameroot = "URL_N_" + str(N) + "_a_" + str(a) + "_beamw_" + str(beamw) + "_" + namestub + "_"
    
    #Saving progenitor point pattern
    np.savetxt("./" + nameroot + "config.txt", flatlat)
    
    #Tiling original point pattern to get all nearest images
    tURL = tile(flatlat,N)
 
    #This set of if/else statements will yield the set of edges for the particular choice of tessellation
    if tess == 'D': 
        ge = np.array(delaunay_graph_2d(tURL)[:,:2], dtype=int)
    elif tess == 'G':
        ge = np.array(gabriel_graph_2d(tURL)[:,:2], dtype=int)
    elif tess == 'V':
        poly = sp.spatial.Voronoi(tURL)
        ge = poly.ridge_vertices
        tURL = poly.vertices
    elif tess == 'C':
        tri = sp.spatial.Delaunay(tURL)
        newpoints = np.zeros([tri.simplices.shape[0],2])
        for i in range(newpoints.shape[0]):
            newpoints[i] = np.mean(tURL[tri.simplices[i]], axis = 0)
        tURL = newpoints
        ge = []
        for i in range(tri.simplices.shape[0]):
            for j in range(3):
                if tri.neighbors[i][j] != -1: ge.append([i,tri.neighbors[i][j]])
        

    sh = beamw/2
    
    #Generating set of rectangle corners for the thickened edges
    beams = []
    for vpair in ge:
        newrect = np.empty([4,2])
        
        if vpair[0] >= 0 and vpair[1] >= 0:
            v0 = tURL[vpair[0]]
            v1 = tURL[vpair[1]]
            rise = (v1[1]-v0[1])
            run = (v1[0]-v0[0])
            th = np.arccos(run/np.linalg.norm(v1-v0))

            if rise < 0: th = -th

            torot = np.array([sh,0])
            c,s = np.cos(th-(np.pi/2)), np.sin(th-(np.pi/2))
            R = np.array(((c,-s),(s,c)))
            newpt = np.dot(R,torot)
            newrect[0] = newpt+v1

            torot = np.array([sh,0])
            c,s = np.cos(th+(np.pi/2)), np.sin(th+(np.pi/2))
            R = np.array(((c,-s),(s,c)))
            newpt = np.dot(R,torot)
            newrect[1] = newpt+v1

            torot = np.array([sh,0])
            c,s = np.cos(th-(np.pi/2)+np.pi), np.sin(th-(np.pi/2)+np.pi)
            R = np.array(((c,-s),(s,c)))
            newpt = np.dot(R,torot)
            newrect[2] = newpt+v0

            torot = np.array([sh,0])
            c,s = np.cos(th+(np.pi/2)+np.pi), np.sin(th+(np.pi/2)+np.pi)
            R = np.array(((c,-s),(s,c)))
            newpt = np.dot(R,torot)
            newrect[3] = newpt+v0

            beams.append(newrect)

    beamhold = []
    for i in beams:
        beamhold.append(Polygon(i))


    #For the Gabriel tessellations, adds a small polygon at each node to prevent gaps in the structure where removed edges would be
    fixhold = []
    if tess == 'G':
        geun, gects = np.unique(ge, return_counts=True)
        needfix = np.where(gects >= 2)[0]
        for f in needfix:
            tedges = np.where(ge==f)[0]
            tfix = f
            nodes = np.unique([ge[tedges]])

            th = 0
            thtest = 0
            nodes = np.delete(nodes, np.where(nodes == tfix))
            onodes = np.array([0,0])
            for i in range(1,nodes.shape[0]):
                for j in range(0,i):
                    v1 = tURL[nodes[i]] - tURL[tfix]
                    v2 = tURL[nodes[j]] - tURL[tfix]

                    nv1 = np.linalg.norm(v1)
                    nv2 = np.linalg.norm(v2)

                    thtest = np.arccos(np.dot(v1,v2)/(nv1*nv2))
                    if thtest > th:
                        th = thtest
                        onodes = [nodes[i],nodes[j]]

            v1 = tURL[onodes[0]] - tURL[tfix]
            v2 = tURL[onodes[1]] - tURL[tfix]

            nv1 = np.linalg.norm(v1)
            nv2 = np.linalg.norm(v2)

            negbis = -(nv2*v1 + nv1*v2)
            negbisfill = sh/np.sin((2*np.pi-th)/2) * negbis/np.linalg.norm(negbis)
            fillpt = tURL[f] + negbisfill

            torot = negbis/np.linalg.norm(negbis) * sh
            c,s = np.cos((np.pi-th)/2), np.sin((np.pi-th)/2)
            R = np.array(((c,-s),(s,c)))
            newpt1 = tURL[f] + np.dot(R,torot)

            torot = negbis/np.linalg.norm(negbis) * sh
            c,s = np.cos(-(np.pi-th)/2), np.sin(-(np.pi-th)/2)
            R = np.array(((c,-s),(s,c)))
            newpt2 = tURL[f] + np.dot(R,torot)

            maybe = tURL[f] + negbis

            fixhold.append(Polygon([tURL[tfix],newpt1,fillpt,newpt2]))


    #Preparations for strucutre pixelization
    plt.gca().set_aspect('equal')

    plt.rcParams["figure.figsize"] = (5,5)

    beamhold = beamhold + fixhold

    packing = shapely.union_all(beamhold+fixhold)

    for i in range(len(beamhold)):

        tx,ty = beamhold[i].exterior.xy

        plt.fill(tx, ty, 'k')

    plt.xlim(0,N)
    plt.ylim(0,N)
    plt.axis('off')

    #dpi can be raised/lowered here to increase/decrease resolution of pixelization
    plt.savefig("./"+nameroot+".png",bbox_inches='tight',pad_inches = 0,dpi=1000)
    return(nameroot,flatlat)


#Sqaure root of the number of the particles in the system
N = int(sys.argv[1])
#Parameter describing the degree of local translational disorder in the URL point patterns
a = float(sys.argv[2])
#Width of the thickened edges in the two-phase material, in units of sqrt(1/number density)
beamw = float(sys.argv[3])
#Specific identifying name for particular simulation
name = str(sys.argv[4])
#Configuration type
ftype = str(sys.argv[5])
#Tessellation type (C,V,G,D)
ntype = str(sys.argv[6])
#Binsize for the angular-averaged spectral density in units of the smallest accessible wavenumber. 
binset = float(sys.argv[7])

#If loading a configuration, the path to the file is loaded here
##Will read in a .txt file whose lines are the global coordinates of each point in the configuration
fn = 0
if ftype == "load": fn = str(sys.argv[8])
   
nameroot, flatlat = networkGen(N, a, beamw, name, fn, tess=ntype, figtype=ftype)

#Loading in pixelized network structures
voxmap = iio.imread("./"+nameroot+".png")[:,:,0]/255

#Setting up and computing the 2D spectral density of the network structure
Lx,Ly = voxmap.shape
phi2 = np.average(voxmap)
nKx,nKy = Lx,Ly
voxmap -= phi2
ccFT = np.fft.fftn(a=voxmap,s=np.array([nKx,nKy]))
packing_bin = []
Xs = np.abs(np.fft.fftshift(ccFT))**2
Kx = np.arange(-nKx//2,nKx//2,dtype=float)
Ky = np.arange(-nKy//2,nKy//2,dtype=float)

Kx *= 2*np.pi/Lx
Ky *= 2*np.pi/Ly

kXgrid, kYgrid = np.meshgrid(Ky,Kx)
mk = np.zeros((kXgrid.shape[0],kYgrid.shape[1]))
comp_mk(kXgrid,kYgrid,mk)

Xs /= (Lx*Ly)**2
Xs *= (mk**2)

#Setting up and computing the angular-averaged spectral density
k_grid = np.zeros((kXgrid.shape[0],kYgrid.shape[0],2))
norms = np.zeros(int((kXgrid.shape[0]*kYgrid.shape[0])),dtype=float)
Sk_norm = np.zeros((int(kXgrid.shape[0]*kYgrid.shape[0]),2))

kXvals = Kx*Lx                                                  
kYvals = Ky*Ly

fill_vecs(Xs, kXvals, kYvals, k_grid, norms, Sk_norm)

unique, counts = np.unique(norms, return_counts=True)

Sk_norm = Sk_norm[Sk_norm[:,1].argsort()]

Sk_norm = np.flip(Sk_norm, axis=1)

bins, binssk = binning_procedure(Sk_norm[1:],binset)

#Wrapping up some normalization
Xv_fin = np.zeros([bins.shape[0],2])

Xv_fin[:,0] = bins/2/np.pi/((flatlat.shape[0])**0.5)

Xv_fin[:,1] = binssk*(flatlat.shape[0])

#Saving only wavenumbers up to a specified cutoff 
Xv_fin = Xv_fin[Xv_fin[:,0] < np.pi*5]

#Exporting binned angular-averaged spectral density
np.savetxt("./"+nameroot+"Xv.txt",Xv_fin)

