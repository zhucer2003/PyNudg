#Initialize solver and construct grid and metric
#------------------------------------------------------------------
#                             Startup.m
# Initialize solver and construct grid and metric
# Purpose : Setup script, building operators, grid, metric and connectivity for 1D solver.     
# Definition of constants

NODETOL = 1e-12
Np = (N+1)*(N+2)/2; Nfp = N+1; Nfaces = 3

# Compute nodal set  
x,y = nudg.Nodes2D(N); r,s = nudg.xytors(x,y); 
# Build reference element matrices
V  = nudg.Vandermonde2D(N, r, s); invV = linalg.inv(V)
MassMatrix = invV.T*invV; 
Dr,Ds = nudg.Dmatrices2D(N, r, s, V);   

# build coordinates of all the nodes
va = num.int32(EToV[:, 0].T); vb =num.int32(EToV[:, 1].T)
vc = num.int32(EToV[:, 2].T)

x = 0.5*(-num.outer(r+s,VX[va])+num.outer(1+r,VX[vb])+num.outer(1+s,VX[vc]));
y = 0.5*(-num.outer(r+s,VY[va])+num.outer(1+r,VY[vb])+num.outer(1+s,VY[vc]));

# find all the nodes that lie on each edge
fmask1   = ( abs(s+1) < NODETOL).nonzero()[0]; 
fmask2   = ( abs(r+s) < NODETOL).nonzero()[0];
fmask3   = ( abs(r+1) < NODETOL).nonzero()[0];
Fmask    = num.vstack((fmask1,fmask2,fmask3))
Fmask  = Fmask.T 
FmaskF = Fmask.reshape(Fmask.shape[0]*Fmask.shape[1],order='F')
Fx = x[FmaskF[:], :]; Fy = y[FmaskF[:], :];

#Create surface integral terms
LIFT = nudg.Lift2D(N,r,s,V,Fmask);

#calculate geometric factors
rx,sx,ry,sy,J = nudg.GeometricFactors2D(x,y,Dr,Ds);

nx, ny, sJ = nudg.Normals2D(Dr,Ds,x,y,K,N,FmaskF);
Fscale = sJ/J[FmaskF,:];
# Build connectivity matrix
EToE, EToF = nudg.Connect2D(EToV);

# Build connectivity maps
mapM, mapP, vmapM, vmapP, vmapB, mapB = nudg.BuildMaps2D(Fmask,VX,VY, EToV, EToE, EToF, K, N, x,y);

#Compute weak operators (could be done in preprocessing to save time)
Vr, Vs = nudg.GradVandermonde2D(N, r, s);
invVV = linalg.inv(num.dot(V,V.T))
Drw = num.dot(num.dot(V,Vr.T),invVV); 
Dsw = num.dot(num.dot(V,Vs.T),invVV);

# get the global variables
G = nudg.Globaldata(N,Nfp,Np,Nv,VX,K,EToV,r,s,x,y,rx,ry,sx,sy,Dr,Ds,LIFT,J,nx,ny,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB,mapM,mapP)

