#------------------------------------------------------------------
#                             Startup.m
# Initialize solver and construct grid and metric
# Purpose : Setup script, building operators, grid, metric and connectivity for 1D solver.     
# Definition of constants

NODETOL = 1e-10
Np = N+1; Nfp = 1; Nfaces = 2

# Compute basic Legendre Gauss Lobatto grid
r = nudg.JacobiGL(0, 0, N)

# Build reference element matrices
V  = nudg.Vandermonde1D(N, r); invV = linalg.inv(V)
Dr = nudg.Dmatrix1D(N, r, V)

# Create surface integral terms
LIFT = nudg.Lift1D(N, r, V)

# build coordinates of all the nodes
va = num.int32(EToV[:, 0].T); vb = num.int32(EToV[:, 1].T)
#print "va",VX
x  = num.outer(num.ones(N+1), VX[va]) + 0.5*num.outer(r+1, VX[vb]-VX[va])

# calculate geometric factors
rx, J = nudg.GeometricFactors1D(x, Dr)

# Compute masks for edge nodes
fmask1 = num.nonzero(abs(r+1) < NODETOL)
fmask2 = (abs(r-1) < NODETOL).nonzero()
Fmask  = num.vstack((fmask1, fmask2))
Fmask  = Fmask.T
Fx     = x[Fmask[:], :]

# Build surface normals and inverse metric at surface
nx = nudg.Normals1D(VX)
Fscale = 1.0/J[Fmask[0, :], :].copy()

# Build connectivity matrix
EToE, EToF = nudg.Connect1D(EToV)

# Build connectivity maps
vmapM, vmapP, vmapB, mapB = nudg.BuildMaps1D(Fmask, EToE, EToF, K, Np, x)

# get the global variables
G = nudg.Globaldata(Np,Nv,VX,K,EToV,r,Dr,LIFT,x,rx,J,nx,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB)


