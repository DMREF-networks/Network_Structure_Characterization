# Network_ELV3D

## Overview
Network_ELV3D computes the edge-length variance of a network structure embedded in three-dimensional euclidean space derived from a spatial tessellation of a point pattern. The point pattern must be in a square simulation box under periodic boundary conditions.

## To run from the command line
python3 Network_ELV3D.py N a simname configtype tesstype (fname)

N - square root of the number of particles in the system
a - Uniformly randomized lattice disorder parameter (leave 0 if not using a URL, otherwise use a > 0)
simname - User-defined string to name a particular calculation
configtype - Three options:
	1 - URL for the uniformly randomized lattiec
	2 - poi for a totally uncorrelated point pattern
	3 - load for using an imported point pattern. Imported files should be a .txt file where the global coordinates of each particle are stored as "x y z" on each line.
tesstype - Four options:
	1 - 'V' for Voronoi
	2 - 'D' for Delaunay
	3 - 'C' for Delaunay Centroidal
	4 - 'G' for Gabriel
fname - If loading in a configuration, this argument is the path to that file relative to Network_ELV3D.py. If not loading a configuration leave this argument blank.

## Yield
Three files:
1 - A .txt file with the generated configuration if using URL or poi
2 - A .npy file with an array of the window radii used in the calculation.
3 - A .npy file with a rectangular matrix where each row contains all of the total edge length values for each window placed in the system. (Thus, the number of columns is the number of windows placed in the system at each value of the window radius.) 

The variance of the rows in the third file are taken after the fact to get the variance at a given window radius. This is done so that the variance can be easily computed across different realizations of the same system. 

