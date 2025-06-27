# Network_2DXv

## Overview
Network_2DXv computes the angular-averaged spectral density of a two-phase network structure derived from a spatial tessellation of a point pattern. The point pattern must be in a square simulation box under periodic boundary conditions.

## To run from the command line
python3 Network_2DXv.py N a beamw simname configtype tesstype binset (fname)

N - square root of the number of particles in the system\
a - Uniformly randomized lattice disorder parameter (leave 0 if not using a URL, otherwise use a > 0)\
beamw - Width of the rectangles from which the two-phase media is generated. Width is in the same units as the box side length.\
simname - User-defined string to name a particular calculation\
configtype - Three options:\
	1 - URL for the uniformly randomized lattice\
	2 - poi for a totally uncorrelated point pattern\
	3 - load for using an imported point pattern. Imported files should be a .txt file where the global coordinates of each particle are stored as "x y" on each line.\
tesstype - Four options:\
	1 - 'V' for Voronoi\
	2 - 'D' for Delaunay\
	3 - 'C' for Delaunay Centroidal\
	4 - 'G' for Gabriel\
binset - Used to control the bin size used in the angular averaging of the spectral density. This number multiplied by the smallest accessible wavenumber will be the bin size.\
fname - If loading in a configuration, this argument is the path to that file relative to Network_2DXv.py. If not loading a configuration leave this argument blank.\

## Yield
Three files:\
1 - A .txt file with the generated configuration if using URL or poi\
2 - A .png of the two-phase material strucutre\
3 - A two-column .txt file where column one contains the wavenumber and column two contains the spectral density values after the angular averaging and binning procedures. 
