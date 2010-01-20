# PyNudg - the python Nodal DG Environment
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

#  2D module


from __future__ import division
from math import sqrt
import copy as cp
import numpy as num

import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.io as io
# 2D parameters
Nfaces=3; NODETOL = 1e-12
eps = num.finfo(float).eps

In = 1; Out = 2; Wall = 3; Far = 4; 
Cyl = 5; Dirichlet = 6; Neuman = 7; Slip = 8;


#Low storage Runge-Kutta coefficients
rk4a = num.array([            0.0 ,\
        -567301805773.0/1357537059087.0,\
        -2404267990393.0/2016746695238.0,\
        -3550918686646.0/2091501179385.0,\
        -1275806237668.0/842570457699.0]);
rk4b = [ 1432997174477.0/9575080441755.0,\
         5161836677717.0/13612068292357.0,\
         1720146321549.0/2090206949498.0 ,\
         3134564353537.0/4481467310338.0 ,\
         2277821191437.0/14882151754819.0];
rk4c = [             0.0  ,\
         1432997174477.0/9575080441755.0 ,\
         2526269341429.0/6820363962896.0 ,\
         2006345519317.0/3224310063776.0 ,\
         2802321613138.0/2924317926251.0];

class Globaldata:
    """ to store the global data that we need to use all the time"""
    def __init__(self,N,Nfp,Np,Nv,VX,K,EToV,r,s,x,y,rx,ry,sx,sy,Dr,Ds,LIFT,J,nx,ny,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB,mapM,mapP):

        self.Np = Np; self.N = N; self.Nfp = Nfp
        self.Nv = Nv; self.VX   = VX; 
        self.K  = K ; self.EToV = EToV;
        self.r  = r ; self.s    = s
        self.Dr = Dr; self.Ds = Ds; self.LIFT = LIFT;
        self.x  = x ; self.rx   = rx;
        self.y  = y ; self.ry   = ry;
        self.sx = sx ; self.sy   = sy;
        self.J  = J ; self.nx   = nx; self.ny = ny
        self.Fscale  = Fscale; self.EToE = EToE;
        self.EToF  = EToF ; self.vmapM = vmapM;
        self.vmapP = vmapP; self.vmapB = vmapB
        self.mapB  = mapB ;self.mapM = mapM
        self.mapP  = mapP
                

    def setglobal(self):
        """function: G=Setglobal(G)
            Purpose:set up the global data"""
        Np = self.Np; N = self.N; Nfp = self.Nfp
        Nv = self.Nv; VX   = self.VX; 
        K  = self.K;  EToV = self.EToV;
        r  = self.r;  s    = self.s
        Dr = self.Dr; Ds   = self.Ds;LIFT = self.LIFT;
        x  = self.x;  y = self.y
        rx   = self.rx; ry   = self.ry;
        sx = self.sx; sy   = self.sy;
        J  = self.J;  nx   = self.nx; ny = self.ny
        Fscale  = self.Fscale; EToE = self.EToE;
        EToF    = self.EToF;  vmapM = self.vmapM;
        vmapP   = self.vmapP; vmapB = self.vmapB
        mapB    = self.mapB; mapM =self.mapM
        mapP    = self.mapP
        return N,Nfp,Np,Nv,VX,K,EToV,r,s,x,y,rx,ry,sx,sy,Dr,Ds,LIFT,J,nx,ny,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB,mapM,mapP

class Ncinfo:
    """store the operator information for Curved elements"""
    def __init__(self,Ncurve):
        self.elmt = []
        self.MM = []
        self.Dx = []
        self.Dy = []
        self.gnx = []
        self.gx = []
        self.gny = []
        self.gy = []
        self.gVM = []
        self.gVP = [] 
        self.glift = [] 




def gamma(z):

    g = 1
    for i in xrange(1, num.int32(z)):
        g = g*i
       
    return g	

def MeshReaderGambit2D(FileName):
    """function [Nv, VX, VY, K, EToV] = MeshReaderGambit2D(FileName)
    Purpose : Read in basic grid information to build gri
    NOTE     : gambit(Fluent, Inc) *.neu format is assumd"""

    Fid = open(FileName, 'r')

    #read after intro
    for i in xrange(6):
        line = Fid.readline()

    # Find number of nodes and number of elements
    dims = Fid.readline().split()
    Nv = num.int(dims[0]); K = num.int32(dims[1]);

    for i in xrange(2):
        line = Fid.readline()

    #read node coordinates
    VX = num.zeros(Nv); VY = num.zeros(Nv);
    for i  in xrange(Nv):
        tmpx = Fid.readline().split()
        VX[i] = float(tmpx[1]); VY[i] = float(tmpx[2]);


    for i in xrange(2):
        line = Fid.readline()

    #read element to node connectivity
    EToV = num.zeros((K, 3));
    for k in xrange(K):
        tmpcon= Fid.readline().split()
        EToV[k,0] = num.int32(tmpcon[3])-1; 
        EToV[k,1] = num.int32(tmpcon[4])-1;
        EToV[k,2] = num.int32(tmpcon[5])-1;

    #Close file
    Fid.close()
    return Nv, VX, VY, K, EToV

def MeshReaderGambitBC2D(FileName):
    """function [Nv, VX, VY, K, EToV, BCType] = MeshReaderGambitBC2D(FileName)
    Purpose  : Read in basic grid information to build grid NOTE     : gambit(Fluent, Inc) *.neu format is assumed"""

    Fid = open(FileName, 'r')

    #read after intro
    for i in xrange(6):
        line = Fid.readline()

    # Find number of nodes and number of elements
    dims = Fid.readline().split()
    Nv = num.int(dims[0]); K = num.int32(dims[1]);

    for i in xrange(2):
        line = Fid.readline()

    #read node coordinates
    VX = num.zeros(Nv); VY = num.zeros(Nv);
    for i  in xrange(Nv):
        tmpx = Fid.readline().split()
        VX[i] = float(tmpx[1]); VY[i] = float(tmpx[2]);


    for i in xrange(2):
        line = Fid.readline()

    #read element to node connectivity
    EToV = num.zeros((K, 3));
    for k in xrange(K):
        tmpcon= Fid.readline().split()
        EToV[k,0] = num.int32(tmpcon[3])-1; 
        EToV[k,1] = num.int32(tmpcon[4])-1;
        EToV[k,2] = num.int32(tmpcon[5])-1;


    # skip through material property section
    for i in xrange(4):
        line = Fid.readline();

    while line.find('ENDOFSECTION')==-1:
        line = Fid.readline()

    line = Fid.readline() 

    # boundary codes (defined in Globals2D)
    BCType = num.zeros((K,3));

    # Read all the boundary conditions at the nodes
    while line != "":
        tmp = Fid.readline().split() 
        if tmp[0] == 'In':  
            bcflag = In
        if tmp[0] == 'Out':
            bcflag = Out
        if tmp[0] == 'Wall':
            bcflag = Wall
        if tmp[0] == 'Far':
            bcflag = Far
        if tmp[0] == 'Cyl':
            bcflag = Cyl
        if tmp[0] == 'Dirichlet':
            bcflag = Dirichlet
        if tmp[0] == 'Neuman':
            bcflag = Neuman    
        if tmp[0] == 'Slip':
            bcflag = Slip
  
        tmp = Fid.readline().split()
        while tmp[0]!='ENDOFSECTION':     
            BCType[num.int32(tmp[0])-1,num.int32(tmp[2])-1] = bcflag
            tmp = Fid.readline().split()

        line = Fid.readline()

    #Close file
    Fid.close();
    return  Nv, VX, VY, K, num.int32(EToV), BCType


def BuildBCMaps2D(Nfp, vmapM, BCType):
    """function BuildMaps2DBC
     Purpose: Build specialized nodal maps for various types of boundary conditions, specified in BCType.""" 

    #create label of face nodes with boundary types from BCType
    bct    = BCType.T;
    bnodes = num.outer(num.ones((Nfp, 1)),\
            bct.reshape(bct.shape[0]*bct.shape[1],1,order='F'));
    bnodes = bnodes.reshape(bnodes.shape[0]*bnodes.shape[1],1,order='F');

    # find location of boundary nodes in face and volume node lists
    mapI = num.nonzero(bnodes==In)[0];
    vmapI = vmapM[mapI];
    mapO = num.nonzero(bnodes==Out)[0]; 
    vmapO = vmapM[mapO];
    mapW = num.nonzero(bnodes==Wall)[0];
    vmapW = vmapM[mapW];
    mapF = num.nonzero(bnodes==Far)[0]; 
    vmapF = vmapM[mapF];
    mapC = num.nonzero(bnodes==Cyl)[0]; 
    vmapC = vmapM[mapC];
    mapD = num.nonzero(bnodes==Dirichlet)[0];
    vmapD = vmapM[mapD];
    mapN = num.nonzero(bnodes==Neuman)[0];
    vmapN = vmapM[mapN];
    mapS = num.nonzero(bnodes==Slip)[0];
    vmapS = vmapM[mapS];
    return mapI, mapO, mapW, mapF, mapC, mapD, mapN, mapS


def MakeCylinder2D(faces, ra, xo, yo, VX, VY,EToV,r,s,x,y,Fmask,FmaskF,N,Dr,Ds,K):
    """Function: MakeCylinder2D(faces, ra, xo, yo)
    Purpose:  Use Gordon-Hall blending with an isoparametric map to modify a list of faces so they conform to a cylinder of radius r centered at (xo,yo)"""

    NCurveFaces = faces.shape[0];
    vflag = num.zeros(VX.shape);
    for n in xrange(NCurveFaces): 

        # move vertices of faces to be curved onto circle
        k = faces[n,0]; f = faces[n,1];
        v1 = EToV[k, f]; v2 = EToV[k,num.mod(f,Nfaces)];
        theta1 = num.arctan2(VY[v1],VX[v1]); 
        theta2 = num.arctan2(VY[v2],VX[v2]);
        newx1 = xo + ra*num.cos(theta1); 
        newy1 = yo + ra*num.sin(theta1);
        newx2 = xo + ra*num.cos(theta2); 
        newy2 = yo + ra*num.sin(theta2);

        #update mesh vertex locations
        VX[v1] = newx1; VX[v2] = newx2; 
        VY[v1] = newy1; VY[v2] = newy2; 

        #store modified vertex numbers
        vflag[v1] = 1;  vflag[v2] = 1;


    # map modified vertex flag to each element
    vflag = vflag[num.int32(EToV)];

    # locate elements with at least one modified vertex
    ks = num.nonzero(num.sum(vflag,1)>0)[0];
    
    # build coordinates of all the corrected nodes
    va = EToV[ks,0]; vb = EToV[ks,1].T; vc = EToV[ks,2].T;
    x[:,ks] = 0.5*(-num.outer(r+s,VX[va])+num.outer(1+r,VX[vb])+num.outer(1+s,VX[vc]));
    y[:,ks] = 0.5*(-num.outer(r+s,VY[va])+num.outer(1+r,VY[vb])+num.outer(1+s,VY[vc]));

    for n in xrange(NCurveFaces):   #deform specified faces
        k = faces[n,0]; f = faces[n,1];

        # find vertex locations for this face and tangential coordinate
        if(f==0): 
            v1 = EToV[k,0]; v2 = EToV[k,1]; vr = r; 
        if(f==1):
            v1 = EToV[k,1]; v2 = EToV[k,2]; vr = s;
        if(f==2): 
            v1 = EToV[k,0]; v2 = EToV[k,2]; vr = s; 
        fr = vr[Fmask[:,f]];
        x1 = VX[v1]; y1 = VY[v1]; 
        x2 = VX[v2]; y2 = VY[v2]

        # move vertices at end points of this face to the cylinder
        theta1 = num.arctan2(y1-yo, x1-xo); theta2 = num.arctan2(y2-yo, x2-xo);

        #check to make sure they are in the same quadrant
        if ((theta2 > 0) and (theta1 < 0)):
            theta1 = theta1 + 2*num.pi; 
        if ((theta1 > 0) and (theta2 < 0)):
            theta2 = theta2 + 2*num.pi; 
  
        #distribute N+1 nodes by arc-length along edge
        theta = 0.5*theta1*(1-fr) + 0.5*theta2*(1+fr);
        theta = theta[:,0]
        #evaluate deformation of coordinates
        fdx = xo + ra*num.cos(theta)-x[Fmask[:,f],k]; 
        fdy = yo + ra*num.sin(theta)-y[Fmask[:,f],k];
        #build 1D Vandermonde matrix for face nodes and volume nodes
        Vface = Vandermonde1D(N, fr); 
        Vvol  = Vandermonde1D(N, vr);
        # compute unblended volume deformations 
        vdx = num.dot(Vvol,num.dot(linalg.inv(Vface),fdx))
        vdy = num.dot(Vvol,num.dot(linalg.inv(Vface),fdy))
        # blend deformation and increment node coordinates
        ids = num.nonzero(abs(1-vr)>1e-7)[0];# warp and blend
        if(f==0): 
            blend = -(r[ids]+s[ids])/(1-vr[ids]); 
        if(f==1): 
            blend =      +(r[ids]+1)/(1-vr[ids]);
        if(f==2): 
            blend = -(r[ids]+s[ids])/(1-vr[ids]); 
        
        x[ids,k] = x[ids,k]+blend.T[0]*vdx[ids];
        y[ids,k] = y[ids,k]+blend.T[0]*vdy[ids];
    
    # repair other coordinate dependent information
    Fx = x[FmaskF, :]; Fy = y[FmaskF, :];
    rx,sx,ry,sy,J = GeometricFactors2D(x, y,Dr,Ds);
    nx, ny, sJ = Normals2D(Dr,Ds,x,y,K,N,FmaskF) 
    Fscale = sJ/J[FmaskF,:];
    
    return VX,VY,x,y,Fx,Fy,rx,ry,sx,sy,nx,ny,sJ, Fscale


def JacobiP(x,alpha,beta,N):

	""" function P = JacobiP(x,alpha,beta,N)
	     Purpose: Evaluate Jacobi Polynomial of type (alpha,beta) > -1
	              (alpha+beta <> -1) at points x for order N and
	              returns P[1:length(xp))]
	     Note   : They are normalized to be orthonormal."""
	N = num.int32(N)
	Nx = len(x)
        if x.shape[0]>1:
            x = x.T
	# Storage for recursive construction
	PL = num.zeros((num.int32(Nx),num.int32(N+1)))

	# Initial values P_0(x) and P_1(x)
	gamma0 = num.power(2.,alpha+beta+1)/(alpha+beta+1.)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1)

	# 
	PL[:,0] = 1.0/sqrt(gamma0)
	if N==0:
            return PL[:,0]

	gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
	PL[:,1] = ((alpha+beta+2)*x/2 + (alpha-beta)/2)/sqrt(gamma1)
	if N==1:
            return PL[:,1]

	# Repeat value in recurrence.
	aold = 2./(2.+alpha+beta)*sqrt((alpha+1.)*(beta+1.)/(alpha+beta+3.))

	# Forward recurrence using the symmetry of the recurrence.
	for i in xrange(1, N):
		h1 = 2.*i+alpha+beta;

		foo = (i+1.)*(i+1.+alpha+beta)*(i+1.+alpha)*(i+1.+beta)/(h1+1.)/(h1+3.)
		anew = 2./(h1+2.)*sqrt(foo);

		bnew = -(alpha*alpha-beta*beta)/(h1*(h1+2.))
		PL[:,i+1] = ( -aold*PL[:,i-1] + num.multiply(x-bnew,PL[:,i]) )/anew
		aold =anew

	return PL[:,N]

def Vandermonde1D(N,xp):

	""" function [V1D] = Vandermonde1D(N,xp)
	    Purpose : Initialize the 1D Vandermonde Matrix.
	 	    V_{ij} = phi_j(xp_i);"""
	
	Nx = num.int32(xp.shape[0])	
	N  = num.int32(N)
	V1D = num.zeros((Nx, N+1))
        
        for j in xrange(N+1):
		V1D[:,j] = JacobiP(xp, 0, 0, j).T # give the tranpose of Jacobi.p

        return V1D

def  JacobiGQ(alpha,beta,N):

        """ function [x,w] = JacobiGQ(alpha,beta,N)
            Purpose: Compute the N'th order Gauss quadrature points, x, 
            and weights, w, associated with the Jacobi 
            polynomial, of type (alpha,beta) > -1 ( <> -0.5)."""

        if N==0: 
            x[0]=(alpha-beta)/(alpha+beta+2)
            w[0] = 2
            return x, w

        # Form symmetric matrix from recurrence.
        J    = num.zeros(N+1)
        h1   = 2*num.arange(N+1) + alpha + beta
        temp = num.arange(N) + 1.0
        J    = num.diag(-1.0/2.0*(alpha**2-beta**2)/(h1+2.0)/h1) + num.diag(2.0/(h1[0:N]+2.0)*num.sqrt(temp*(temp+alpha+beta)*(temp+alpha)*(temp+beta)/(h1[0:N]+1.0)/(h1[0:N]+3.0)),1)
    
        if alpha+beta < 10*num.finfo(float).eps : 
            J[0,0] = 0.0
        J = J + J.T
    
        #Compute quadrature by eigenvalue solve
        D,V = linalg.eig(J)
        ind = num.argsort(D)
        D = D[ind]
        V = V[:,ind]
        x = D
        w = (V[0,:].T)**2*2**(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1)

        return x, w

def  JacobiGL(alpha,beta,N):

        """ function [x] = JacobiGL(alpha,beta,N)
             Purpose: Compute the Nth order Gauss Lobatto quadrature points, x, associated with the Jacobi polynomia           l,of type (alpha,beta) > -1 ( <> -0.5).""" 
    
        x = num.zeros((N+1,1))
        if N==1:
            x[0]=-1.0
            x[1]=1.0
            return x
    
        xint,w = JacobiGQ(alpha+1,beta+1,N-2)
    
        x = num.hstack((-1.0,xint,1.0))
    
        return x.T



def Warpfactor(N, rout):
    """function warp = Warpfactor(N, rout)
       Purpose  : Compute scaled warp function at order N based on rout interpolation nodes"""

    # Compute LGL and equidistant node distribution
    LGLr = JacobiGL(0,0,N); req  = num.linspace(-1,1,N+1)
    # Compute V based on req
    Veq = Vandermonde1D(N,req);
    # Evaluate Lagrange polynomial at rout
    Nr = len(rout); Pmat = num.zeros((N+1,Nr));
    for i in xrange(N+1):  
        Pmat[i,:] = JacobiP(rout.T[0,:], 0, 0, i)
    
    Lmat = linalg.solve(Veq.T,Pmat)

    # Compute warp factor
    warp = num.dot(Lmat.T,LGLr - req);
    warp = warp.reshape(Lmat.shape[1],1)
    zerof = (abs(rout)<1.0-1.0e-10)
    sf = 1.0 - (zerof*rout)**2
    warp = warp/sf + warp*(zerof-1)
    return warp

def Nodes2D(N):
    """function [x,y] = Nodes2D(N);
       Purpose  : Compute (x,y) nodes in equilateral triangle for polynomial of order N"""            

    alpopt = num.array([0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,\
            1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223,1.6258]);
          
    # Set optimized parameter, alpha, depending on order N
    if N< 16:
        alpha = alpopt[N-1]
    else:
        alpha = 5.0/3.0


    # total number of nodes
    Np = (N+1)*(N+2)/2.0

    # Create equidistributed nodes on equilateral triangle
    L1 = num.zeros((Np,1)); L2 = num.zeros((Np,1)); L3 = num.zeros((Np,1));
    sk = 0;
    for n in xrange(N+1):
        for m in xrange(N+1-n):
            L1[sk] = n/N; 
            L3[sk] = m/N
            sk = sk+1
  
    L2 = 1.0-L1-L3 
    x = -L2+L3; y = (-L2-L3+2*L1)/sqrt(3.0);
    
    # Compute blending function at each node for each edge
    blend1 = 4*L2*L3; blend2 = 4*L1*L3; blend3 = 4*L1*L2
   
    # Amount of warp for each node, for each edge
    warpf1 = Warpfactor(N,L3-L2); 
    warpf2 = Warpfactor(N,L1-L3); 
    warpf3 = Warpfactor(N,L2-L1);
   
    # Combine blend & warp
    warp1 = blend1*warpf1*(1 + (alpha*L1)**2);
    warp2 = blend2*warpf2*(1 + (alpha*L2)**2);
    warp3 = blend3*warpf3*(1 + (alpha*L3)**2);
    # Accumulate deformations associated with each edge
    x = x + 1*warp1 + num.cos(2*num.pi/3)*warp2 + num.cos(4*num.pi/3)*warp3;
    y = y + 0*warp1 + num.sin(2*num.pi/3)*warp2 + num.sin(4*num.pi/3)*warp3;
    return x,y

def xytors(x,y):
    """function [r,s] = xytors(x, y)
    Purpose : From (x,y) in equilateral triangle to (r,s) coordinates in standard triangle"""

    L1 = (num.sqrt(3.0)*y+1.0)/3.0;
    L2 = (-3.0*x - num.sqrt(3.0)*y + 2.0)/6.0;
    L3 = ( 3.0*x - num.sqrt(3.0)*y + 2.0)/6.0;

    r = -L2 + L3 - L1; s = -L2 - L3 + L1;
    return r,s

def rstoab(r,s):
    """function [a,b] = rstoab(r,s)
 Purpose : Transfer from (r,s) -> (a,b) coordinates in triangle"""

    Np = len(r); a = num.zeros((Np,1));
    for n in xrange(Np):
        if s[n] != 1:
            a[n] = 2*(1+r[n])/(1-s[n])-1;
        else:
            a[n] = -1

    b = s.reshape(-1,1)
    return a,b

def Simplex2DP(a,b,i,j):
    """function [P] = Simplex2DP(a,b,i,j);
     Purpose : Evaluate 2D orthonormal polynomial
           on simplex at (a,b) of order (i,j)"""
    h1 = JacobiP(a,0,0,i).reshape(len(a),1);
    h2 = JacobiP(b,2*i+1,0,j).reshape(len(a),1)
    P  = num.sqrt(2.0)*h1*h2*(1-b)**i
    return P[:,0]

def Vandermonde2D(N, r, s):
    """function [V2D] = Vandermonde2D(N, r, s);
    Purpose : Initialize the 2D Vandermonde Matrix,  V_{ij} = phi_j(r_i, s_i)"""

    V2D = num.zeros((len(r),(N+1)*(N+2)/2))

    # Transfer to (a,b) coordinates
    a, b = rstoab(r, s);
    
    # build the Vandermonde matrix
    sk = 0;

    for i in xrange(N+1):
        for j in xrange(N-i+1):
            V2D[:,sk] = Simplex2DP(a,b,i,j)
            sk = sk+1;
    return V2D

def GradJacobiP(z, alpha, beta, N):
    """ function [dP] = GradJacobiP(z, alpha, beta, N);
    Purpose: Evaluate the derivative of the orthonormal Jacobi polynomial of type (alpha,beta)>-1, at points x for order N and returns dP[1:length(xp))]"""
    Nx = num.int32(z.shape[0])
    dP = num.zeros((Nx, 1))
    if N==0:
	dP[:,0] = 0.0
    else:	
        dP[:,0]= sqrt(N*(N+alpha+beta+1.))*JacobiP(z,alpha+1,beta+1, N-1)

    return dP

def GradSimplex2DP(a,b,id,jd):
    """function [dmodedr, dmodeds] = GradSimplex2DP(a,b,id,jd)
    Purpose: Return the derivatives of the modal basis (id,jd) on the 2D simplex at (a,b)."""   
    fa  = JacobiP(a, 0, 0, id).reshape(len(a),1); 
    dfa = GradJacobiP(a, 0, 0, id);
    gb  = JacobiP(b, 2*id+1,0, jd).reshape(len(b),1) 
    dgb = GradJacobiP(b, 2*id+1,0, jd)
    
    # r-derivative
    # d/dr = da/dr d/da + db/dr d/db = (2/(1-s)) d/da = (2/(1-b)) d/da
    dmodedr = dfa*gb;
    if(id>0):
        dmodedr = dmodedr*((0.5*(1-b))**(id-1));

    # s-derivative
    # d/ds = ((1+a)/2)/((1-b)/2) d/da + d/db
    dmodeds = dfa*(gb*(0.5*(1+a)));
    if(id>0):
        dmodeds = dmodeds*((0.5*(1-b))**(id-1));
    tmp = dgb*((0.5*(1-b))**id);
    if(id>0):
        tmp = tmp-0.5*id*gb*((0.5*(1-b))**(id-1));
    dmodeds = dmodeds+fa*tmp;
    # Normalize
    dmodedr = 2**(id+0.5)*dmodedr;  
    dmodeds = 2**(id+0.5)*dmodeds; 
   
    return dmodedr[:,0], dmodeds[:,0]


def GradVandermonde2D(N,r,s):
    """function [V2Dr,V2Ds] = GradVandermonde2D(N,r,s)
    Purpose : Initialize the gradient of the modal basis (i,j) at (r,s) at order N"""	

    V2Dr = num.zeros((len(r),(N+1)*(N+2)/2));
    V2Ds = num.zeros((len(r),(N+1)*(N+2)/2));

    # find tensor-product coordinates
    a,b = rstoab(r,s);
    # Initialize matrices
    sk = 0;
    for i in xrange(N+1):
        for j in xrange(N-i+1):
            V2Dr[:,sk],V2Ds[:,sk] = GradSimplex2DP(a,b,i,j);
            sk = sk+1;
    return V2Dr,V2Ds

def Dmatrices2D(N,r,s,V):
    """function [Dr,Ds] = Dmatrices2D(N,r,s,V)
    Purpose : Initialize the (r,s) differentiation matriceon the simplex, evaluated at (r,s) at order N"""
    Vr, Vs = GradVandermonde2D(N, r, s);
    invV   = linalg.inv(V) 
    Dr     = num.dot(Vr,invV); 
    Ds     = num.dot(Vs,invV);
    return Dr,Ds

def Lift2D(N,r,s,V,Fmask):
    """function [LIFT] = Lift2D()
    Purpose  : Compute surface to volume lift term for DG formulation"""
    Nfp = N+1; Np = (N+1)*(N+2)/2; 
    Emat = num.zeros((Np, Nfaces*Nfp));

    # face 1
    faceR = r[Fmask[:,0]];
    V1D = Vandermonde1D(N, faceR); 
    massEdge1 = linalg.inv(num.dot(V1D,V1D.T));
    Emat[Fmask[:,0],0:Nfp] = massEdge1;

    # face 2
    faceR = r[Fmask[:,1]];
    V1D = Vandermonde1D(N, faceR);
    massEdge2 = linalg.inv(num.dot(V1D,V1D.T));
    Emat[Fmask[:,1],Nfp:2*Nfp] = massEdge2;

    # face 3
    faceS = s[Fmask[:,2]];
    V1D = Vandermonde1D(N, faceS); 
    massEdge3 = linalg.inv(num.dot(V1D,V1D.T));
    Emat[Fmask[:,2],2*Nfp:3*Nfp] = massEdge3;

    # inv(mass matrix)*\I_n (L_i,L_j)_{edge_n}
    LIFT = num.dot(V,num.dot(V.T,Emat));
    return LIFT

def GeometricFactors2D(x,y,Dr,Ds):
    """function [rx,sx,ry,sy,J] = GeometricFactors2D(x,y,Dr,Ds)
    Purpose  : Compute the metric elements for the local mappings of the elements"""
    #Calculate geometric factors
    xr = num.dot(Dr,x); xs = num.dot(Ds,x); 
    yr = num.dot(Dr,y); ys = num.dot(Ds,y); J = -xs*yr + xr*ys;
    rx = ys/J; sx =-yr/J; ry =-xs/J; sy = xr/J;
    return rx, sx, ry, sy, J

def Normals2D(Dr,Ds,x,y,K,N,Fmask):
    """function [nx, ny, sJ] = Normals2D()
    #Purpose : Compute outward pointing normals at elements faces and surface Jacobians"""
    Nfp = N+1; Np = (N+1)*(N+2)/2; 
    xr = num.dot(Dr,x); yr = num.dot(Dr,y); 
    xs = num.dot(Ds,x); ys = num.dot(Ds,y); J = xr*ys-xs*yr;

    # interpolate geometric factors to face nodes
    fxr = xr[Fmask, :]; fxs = xs[Fmask, :]; 
    fyr = yr[Fmask, :]; fys = ys[Fmask, :];

    # build normals
    nx = num.zeros((3*Nfp, K)); ny = num.zeros((3*Nfp, K))
    fid1 = num.arange(Nfp).reshape(Nfp,1); 
    fid2 = fid1+Nfp; 
    fid3 = fid2+Nfp;

    # face 1

    nx[fid1, :] =  fyr[fid1, :];
    ny[fid1, :] = -fxr[fid1, :];

    # face 2
    nx[fid2, :] =  fys[fid2, :]-fyr[fid2, :]; 
    ny[fid2, :] = -fxs[fid2, :]+fxr[fid2, :];

    # face 3
    nx[fid3, :] = -fys[fid3, :]; 
    ny[fid3, :] =  fxs[fid3, :];

    # normalise
    sJ = num.sqrt(nx*nx+ny*ny); 
    nx = nx/sJ; ny = ny/sJ;
    return nx, ny, sJ

def Connect2D(EToV):
    """function [EToE, EToF] = Connect2D(EToV)
    Purpose  : Build global connectivity arrays for grid based on standard EToV input array from grid generator"""
    Nfaces = 3;
    # Find number of elements and vertices
    K = EToV.shape[0]; Nv = EToV.max()+1;

    #Create face to node connectivity matrix
    TotalFaces = Nfaces*K;

    # List of local face to local vertex connections
    vn = num.int32([[0,1],[1,2],[0,2]]);
    # Build global face to node sparse array
    SpFToV = sparse.lil_matrix((TotalFaces, Nv));
    sk = 0;
    for k in xrange(K):
        for face in range(Nfaces):
            SpFToV[sk, EToV[k, vn[face,:]]] = 1;
            sk = sk+1;

    # Build global face to global face sparse array
    speye = sparse.lil_matrix((TotalFaces,TotalFaces))
    speye.setdiag(num.ones(TotalFaces))
    SpFToF = num.dot(SpFToV,SpFToV.T) - 2*speye;

    # Find complete face to face connections
    a = num.array(num.nonzero(SpFToF.todense()==2))
    faces1 = a[0,:].T
    faces2 = a[1,:].T
    del a
    #Convert faceglobal number to element and face numbers
    element1 = num.floor( (faces1)/Nfaces )  ; 
    face1    = num.mod( (faces1), Nfaces ) ;
    element2 = num.floor( (faces2)/Nfaces )  ; 
    face2    = num.mod( (faces2), Nfaces ) ;

    #Rearrange into Nelements x Nfaces sized arrays
    size = num.array([K,Nfaces]) 
    ind = sub2ind([K, Nfaces], element1, face1);

    EToE = num.outer(num.arange(K),num.ones((1,Nfaces)))
    EToF = num.outer(num.ones((K,1)),num.arange(Nfaces))
    EToE = EToE.reshape(K*Nfaces)   
    EToF = EToF.reshape(K*Nfaces) 

    
    EToE[num.int32(ind)] = element2.copy()
    EToF[num.int32(ind)] = face2.copy()
    
    EToE = EToE.reshape(K,Nfaces)
    EToF = EToF.reshape(K,Nfaces)
    
    return  EToE, EToF 
 
def sub2ind(size,I,J):
    """function: IND = sub2ind(size,I,J) 
    Purpose:returns the linear index equivalent to the row and column subscripts I and J for a matrix of size siz. siz is a vector with ndim(A) elements (in this case, 2), where siz(1) is the number of rows and siz(2) is the number of columns."""
    ind = I*size[1]+J
    return ind

def BuildMaps2D(Fmask,VX,VY, EToV, EToE, EToF, K, N, x,y):
    """function [mapM, mapP, vmapM, vmapP, vmapB, mapB] = BuildMaps2D
    Purpose: Connectivity and boundary tables in the K # of Np elements"""
    Nfp = N+1; Np = (N+1)*(N+2)/2; 
    #number volume nodes consecutively
    temp    = num.arange(K*Np)
    nodeids = temp.reshape(Np, K,order='F').copy();
    
    vmapM   = num.zeros((Nfp, Nfaces, K)); 
    vmapP   = num.zeros((Nfp, Nfaces, K)); 
    mapM    = num.arange(num.int32(K)*Nfp*Nfaces);    
    mapP    = mapM.reshape(Nfp, Nfaces, K).copy();
    # find index of face nodes with respect to volume node ordering
    for k1 in xrange(K):
        for f1 in range(Nfaces):
            vmapM[:,f1,k1] = nodeids[Fmask[:,f1], k1];
    
    # need to figure it out
    xtemp = x.reshape(K*Np,1,order='F').copy() 
    ytemp = y.reshape(K*Np,1,order='F').copy() 

    one = num.ones((1,Nfp));
    for k1 in xrange(K):
        for f1 in xrange(Nfaces):
            # find neighbor
            k2 = EToE[k1,f1]; f2 = EToF[k1,f1];
    
            # reference length of edge
            v1 = EToV[k1,f1]; 
            v2 = EToV[k1, 1+num.mod(f1,Nfaces-1)];
            
            refd = num.sqrt((VX[v1]-VX[v2])**2 \
                    + (VY[v1]-VY[v2])**2 );
            #find find volume node numbers of left and right nodes 
            vidM = vmapM[:,f1,k1]; vidP = vmapM[:,f2,k2];
            x1 = xtemp[num.int32(vidM)]; 
            y1 = ytemp[num.int32(vidM)];
            x2 = xtemp[num.int32(vidP)];
            y2 = ytemp[num.int32(vidP)];
            x1 = num.dot(x1,one);  y1 = num.dot(y1,one);  
            x2 = num.dot(x2,one);  y2 = num.dot(y2,one);
            #Compute distance matrix
            D = (x1 -x2.T)**2 + (y1-y2.T)**2;
            # need to figure it out
            idM, idP = num.nonzero(num.sqrt(abs(D))<NODETOL*refd);
            vmapP[idM,f1,k1] = vidP[idP]; 
            mapP[idM,f1,k1] = idP + f2*Nfp+k2*Nfaces*Nfp;

    # reshape vmapM and vmapP to be vectors and create boundary node list
    vmapP = vmapP.reshape(Nfp*Nfaces*K,1,order='F'); 
    vmapM = vmapM.reshape(Nfp*Nfaces*K,1,order='F');
    mapP  = mapP.reshape(Nfp*Nfaces*K,1,order='F'); 
    mapB  = num.array((vmapP==vmapM).nonzero())[0,:] 
    mapB  = mapB.reshape(len(mapB),1)
    vmapB = vmapM[mapB].reshape(len(mapB),1);
    return num.int32(mapM), num.int32(mapP), num.int32(vmapM), num.int32(vmapP), num.int32(vmapB), num.int32(mapB)


def dtscale2D(r,s,x,y):
    """function dtscale = dtscale2D;
    Purpose : Compute inscribed circle diameter as characteristic for grid to choose timestep"""

    #Find vertex nodes
    vmask1   = (abs(s+r+2) < NODETOL).nonzero()[0]; 
    vmask2   = (abs(r-1)   < NODETOL).nonzero()[0];
    vmask3   = (abs(s-1)   < NODETOL).nonzero()[0];
    vmask    = num.vstack((vmask1,vmask2,vmask3));
    vmask    = vmask.T
    vmaskF   = vmask.reshape(vmask.shape[0]*vmask.shape[1],order='F')

    vx = x[vmaskF[:], :]; vy = y[vmaskF[:], :];

    #Compute semi-perimeter and area
    len1 = num.sqrt((vx[0,:]-vx[1,:])**2\
            +(vy[0,:]-vy[1,:])**2);
    len2 = num.sqrt((vx[1,:]-vx[2,:])**2\
            +(vy[1,:]-vy[2,:])**2);
    len3 = num.sqrt((vx[2,:]-vx[0,:])**2\
            +(vy[2,:]-vy[0,:])**2);
    sper = (len1 + len2 + len3)/2.0; 
    Area = num.sqrt(sper*(sper-len1)*(sper-len2)*(sper-len3));

    # Compute scale using radius of inscribed circle
    dtscale = Area/sper;
    return dtscale

def Maxwell2D(Hx, Hy, Ez, FinalTime,G):
    """function [Hx,Hy,Ez] = Maxwell2D(Hx, Hy, Ez, FinalTime)
    Purpose :Integrate TM-mode Maxwell's until FinalTime starting with initial conditions Hx,Hy,Ez"""
    # set up the parameters
    N,Nfp,Np,Nv,VX,K,EToV,r,s,x,y,rx,ry,sx,sy,Dr,Ds,LIFT,J,nx,ny,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB,mapM,mapP=G.setglobal()   
    
    time = 0;

    # Runge-Kutta residual storage  
    resHx = num.zeros((Np,K)); 
    resHy = num.zeros((Np,K)); 
    resEz = num.zeros((Np,K)); 

    #compute time step size
    rLGL = JacobiGQ(0,0,N)[0]; rmin = abs(rLGL[0]-rLGL[1]);
    dtscale = dtscale2D(r,s,x,y); dt = dtscale.min()*rmin*2/3
    
    #outer time step loop 
    while (time<FinalTime):
  
        if(time+dt>FinalTime):
            dt = FinalTime-time
        
        for INTRK in xrange(5):
            #compute right hand side of TM-mode Maxwell's equations
            rhsHx, rhsHy, rhsEz = MaxwellRHS2D(Hx,Hy,Ez,G)
            
            #initiate and increment Runge-Kutta residuals
            resHx = rk4a[INTRK]*resHx + dt*rhsHx;  
            resHy = rk4a[INTRK]*resHy + dt*rhsHy; 
            resEz = rk4a[INTRK]*resEz + dt*rhsEz; 
            
            #update fields
            Hx = Hx+rk4b[INTRK]*resHx;
            Hy = Hy+rk4b[INTRK]*resHy; 
            Ez = Ez+rk4b[INTRK]*resEz;   
        # Increment time
        time = time+dt;
    
    return Hx, Hy, Ez, time



def MaxwellRHS2D(Hx,Hy,Ez,G):
    """function [rhsHx, rhsHy, rhsEz] = MaxwellRHS2D(Hx,Hy,Ez)
    Purpose  : Evaluate RHS flux in 2D Maxwell TM form""" 
    # set up the parameters
    N,Nfp,Np,Nv,VX,K,EToV,r,s,x,y,rx,ry,sx,sy,Dr,Ds,LIFT,J,nx,ny,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB,mapM,mapP=G.setglobal()

    # Define field differences at faces
    vmapM = vmapM.reshape(Nfp*Nfaces,K,order='F')
    vmapP = vmapP.reshape(Nfp*Nfaces,K,order='F')
    Im,Jm = ind2sub(vmapM,Np)
    Ip,Jp = ind2sub(vmapP,Np)
    
    dHx = num.zeros((Nfp*Nfaces,K));
    dHx = Hx[Im,Jm]-Hx[Ip,Jp];
    dHy = num.zeros((Nfp*Nfaces,K)); 
    dHy = Hy[Im,Jm]-Hy[Ip,Jp]; 
    dEz = num.zeros((Nfp*Nfaces,K)); 
    dEz = Ez[Im,Jm]-Ez[Ip,Jp];

    # Impose reflective boundary conditions (Ez+ = -Ez-)
    size_H= Nfp*Nfaces
    I,J = ind2sub(mapB,size_H)
    Iz,Jz = ind2sub(vmapB,Np)

    dHx[I,J] = 0; dHy[I,J] = 0; 
    dEz[I,J] = 2*Ez[Iz,Jz];
    #evaluate upwind fluxes
    alpha  = 1.0;
    ndotdH =  nx*dHx + ny*dHy;
    fluxHx =  ny*dEz + alpha*(ndotdH*nx-dHx);
    fluxHy = -nx*dEz + alpha*(ndotdH*ny-dHy);
    fluxEz = -nx*dHy + ny*dHx - alpha*dEz;
    
    #local derivatives of fields
    Ezx,Ezy = Grad2D(Ez,Dr,Ds,rx,ry,sx,sy); CuHx,CuHy,CuHz = Curl2D(Hx,Hy,0,Dr,Ds,rx,ry,sx,sy);
    #compute right hand sides of the PDE's
    rhsHx = -Ezy  + num.dot(LIFT,Fscale*fluxHx)/2.0;
    rhsHy =  Ezx  + num.dot(LIFT,Fscale*fluxHy)/2.0;
    rhsEz =  CuHz + num.dot(LIFT,Fscale*fluxEz)/2.0;
    return rhsHx, rhsHy, rhsEz

def ind2sub(matr,row_size):
    """purpose: convert linear index to 2D index""" 
    I = num.int32(num.mod(matr,row_size))
    J = num.int32((matr - I)/row_size)
    return I,J

def Grad2D(u,Dr,Ds,rx,ry,sx,sy):
    """function [ux,uy] = Grad2D(u);
    Purpose: Compute 2D gradient field of scalar u"""
    ur = num.dot(Dr,u); us = num.dot(Ds,u);
    ux = rx*ur + sx*us; 
    uy = ry*ur + sy*us
    return  ux, uy

def Curl2D(ux,uy,uz,Dr,Ds,rx,ry,sx,sy):
    """function [vx,vy,vz] = Curl2D(ux,uy,uz);
    Purpose: Compute 2D curl-operator in (x,y) plane"""
    uxr = num.dot(Dr,ux); 
    uxs = num.dot(Ds,ux);
    uyr = num.dot(Dr,uy); 
    uys = num.dot(Ds,uy); 
    vz =  rx*uyr + sx*uys\
            - ry*uxr - sy*uxs;
    vx = 0; vy = 0
    if (uz!=0):
        uzr = num.dot(Dr,uz); uzs = num.dot(Ds,uz);
        vx =  ry*uzr + sy*uzs; 
        vy = -rx*uzr - sx*uzs;
    return vx, vy, vz

def MaxwellCurved2D(Hx, Hy, Ez, FinalTime,G,V,curved):
    """function [Hx,Hy,Ez] = Maxwell2D(Hx, Hy, Ez, FinalTime)
    Purpose :Integrate TM-mode Maxwell's until FinalTime starting with initial conditions Hx,Hy,Ez"""
    # set up the parameters
    N,Nfp,Np,Nv,VX,K,EToV,r,s,x,y,rx,ry,sx,sy,Dr,Ds,LIFT,J,nx,ny,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB,mapM,mapP=G.setglobal()   
    
    time = 0;

    # Runge-Kutta residual storage  
    resHx = num.zeros((Np,K)); 
    resHy = num.zeros((Np,K)); 
    resEz = num.zeros((Np,K)); 

    #compute time step size
    rLGL = JacobiGQ(0,0,N)[0]; rmin = abs(rLGL[0]-rLGL[1]);
    dtscale = dtscale2D(r,s,x,y); dt = dtscale.min()*rmin*2/3
    
    cinfo = BuildCurvedOPS2D(3*N,G,V,curved)

    
    #outer time step loop 
    while (time<FinalTime):
  
        if(time+dt>FinalTime):
            dt = FinalTime-time
        
        for INTRK in xrange(5):
            #compute right hand side of TM-mode Maxwell's equations
            rhsHx, rhsHy, rhsEz = MaxwellCurvedRHS2D(cinfo,Hx,Hy,Ez,G)
            
            #initiate and increment Runge-Kutta residuals
            resHx = rk4a[INTRK]*resHx + dt*rhsHx;  
            resHy = rk4a[INTRK]*resHy + dt*rhsHy; 
            resEz = rk4a[INTRK]*resEz + dt*rhsEz; 
            
            #update fields
            Hx = Hx+rk4b[INTRK]*resHx;
            Hy = Hy+rk4b[INTRK]*resHy; 
            Ez = Ez+rk4b[INTRK]*resEz;   
        # Increment time
        time = time+dt;
    
    return Hx, Hy, Ez, time

def MaxwellCurvedRHS2D(cinfo, Hx,Hy,Ez,G):
    """function [rhsHx, rhsHy, rhsEz] = MaxwellCurvedRHS2D(cinfo, Hx,Hy,Ez)
    Purpose  : Evaluate RHS flux in 2D Maxwell TM form""" 
    N,Nfp,Np,Nv,VX,K,EToV,r,s,x,y,rx,ry,sx,sy,Dr,Ds,LIFT,J,nx,ny,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB,mapM,mapP=G.setglobal()   

    rhsHx,rhsHy,rhsEz = MaxwellRHS2D(Hx, Hy, Ez,G);
    
    # correct residuals at each curved element
    Ncinfo = len(cinfo);
    for n in xrange(Ncinfo):
        #for each curved element computed L2 derivatives via cubature
        cur = cinfo[n]; k1 = cur.elmt; cDx = cur.Dx; cDy = cur.Dy; 

        rhsHx[:,k1] = -num.dot(cDy,Ez[:,k1]);
        rhsHy[:,k1] =  num.dot(cDx,Ez[:,k1]);
        rhsEz[:,k1] =  num.dot(cDx,Hy[:,k1]) - num.dot(cDy,Hx[:,k1]);
    
        # for each face of each curved element use Gauss quadrature based lifts
        for f1 in xrange(Nfaces):
            k2 = EToE[k1,f1];
            gnx = cur.gnx[:,f1]; gny = cur.gny[:,f1];
            gVM = cur.gVM[:,:,f1]; gVP = cur.gVP[:,:,f1];
            glift = cur.glift[:,:,f1];

            #compute difference of solution traces at Gauss nodes
            gdHx = num.dot(gVM,Hx[:,k1]) - num.dot(gVP,Hx[:,k2]);
            gdHy = num.dot(gVM,Hy[:,k1]) - num.dot(gVP,Hy[:,k2]);
            gdEz = num.dot(gVM,Ez[:,k1]) - num.dot(gVP,Ez[:,k2]);

            #correct jump at Gauss nodes on domain boundary faces
            if(k1==k2):
                gdHx = 0*gdHx; gdHy = 0*gdHy; gdEz = 2*num.dot(gVM,Ez[:,k1]);
    

            # perform upwinding
            gndotdH =  gnx*gdHx+gny*gdHy;
            fluxHx =  gny*gdEz + gndotdH*gnx-gdHx;
            fluxHy = -gnx*gdEz + gndotdH*gny-gdHy;
            fluxEz = -gnx*gdHy + gny*gdHx   -gdEz;

            #lift flux terms using Gauss based lift operator
            rhsHx[:,k1] = rhsHx[:,k1] + num.dot(glift,fluxHx)/2;
            rhsHy[:,k1] = rhsHy[:,k1] + num.dot(glift,fluxHy)/2;
            rhsEz[:,k1] = rhsEz[:,k1] + num.dot(glift,fluxEz)/2;
  
    return rhsHx,rhsHy,rhsEz

def BuildCurvedOPS2D(intN,G,V,curved):
    """function [cinfo] = BuildCurvedOPS2D(intN)
    Purpose: build curved info for curvilinear elements"""
    N,Nfp,Np,Nv,VX,K,EToV,r,s,x,y,rx,ry,sx,sy,Dr,Ds,LIFT,J,nx,ny,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB,mapM,mapP=G.setglobal()   
 
    # 1. Create cubature information

    # 1.1 Extract cubature nodes and weights
    cR,cS,cW,Ncub = Cubature2D(intN);
    
    # 1.1. Build interpolation matrix (nodes->cubature nodes)
    cV = InterpMatrix2D(N,V,cR, cS);

    # 1.2 Evaluate derivatives of Lagrange interpolants at cubature nodes
    cDr,cDs = Dmatrices2D(N, cR, cS, V);
    
    # 2. Create surface quadrature information

    # 2.1 Compute Gauss nodes and weights for 1D integrals
    gz, gw = JacobiGQ(0, 0, intN);
  

    # 2.2 Build Gauss nodes running counter-clockwise on element faces
    gR = num.vstack(((gz, -gz), -num.ones(gz.shape)));
    gS = num.vstack(((-num.ones(gz.shape), gz), -gz));
    gz = gz.reshape(-1,1)
    gw = gw.reshape(-1,1)
    gR = gR.T
    gS = gS.T

    # 2.3 For each face
    gV  = num.zeros((gR.shape[0],V.shape[0],Nfaces))
    gDr = num.zeros((gR.shape[0],V.shape[0],Nfaces)) 
    gDs = num.zeros((gR.shape[0],V.shape[0],Nfaces)) 

    for f1 in xrange(Nfaces):
        #2.3.1 build nodes->Gauss quadrature interpolation and differentiation matrices
        gV[:,:,f1] = InterpMatrix2D(N,V,gR[:,f1], gS[:,f1]);
        gDr[:,:,f1],gDs[:,:,f1] = Dmatrices2D(N, gR[:,f1], gS[:,f1], V);


    # 3. For each curved element, evaluate custom operator matrices
    Ncurved = len(curved);

    # 3.1 Store custom information in array of Matlab structs
    cinfo = Ncinfo(Ncurved)
    test  = [];
    for c in xrange(Ncurved): 
        #find next curved element and the coordinates of its nodes
        k1 = curved[c]; x1 = x[:,k1]; y1 = y[:,k1]; cinfo.elmt=k1

        #compute geometric factors
        crx,csx,cry,csy,cJ = GeometricFactors2D(x1,y1,cDr,cDs);

        # build mass matrix
        cMM = num.dot(num.dot(cV.T,num.diag(cJ*cW)),cV);
        cinfo.MM = cMM

        # build physical derivative matrices
        cinfo.Dx = num.dot(linalg.inv(cMM),\
                num.dot(num.dot(cV.T,num.diag(cW*cJ)),(num.dot(num.diag(crx),cDr) + num.dot(num.diag(csx),cDs))));
        cinfo.Dy= num.dot(linalg.inv(cMM),\
                num.dot(num.dot(cV.T,num.diag(cW*cJ)),(num.dot(num.diag(cry),cDr) + num.dot(num.diag(csy),cDs))));

        # build individual lift matrices at each face
        temp_gnx = num.zeros((gDr.shape[0],Nfaces))
        temp_gny = num.zeros((gDr.shape[0],Nfaces))
        temp_gx = num.zeros((gDr.shape[0],Nfaces))
        temp_gy = num.zeros((gDr.shape[0],Nfaces))
        temp_gVM = num.zeros((gV.shape[0],gV.shape[1],Nfaces))
        temp_gVP = num.zeros((gV.shape[0],gV.shape[1],Nfaces))
        temp_glift = num.zeros((gDr.shape[1],gDr.shape[0],Nfaces))

        for f1 in xrange(Nfaces):
            k2 = EToE[k1,f1]; f2 = EToF[k1,f1];

            #compute geometric factors
            grx,gsx,gry,gsy,gJ = GeometricFactors2D(x1,y1,gDr[:,:,f1],gDs[:,:,f1]);
            grx,gsx,gry,gsy,gJ = grx.reshape(-1,1),gsx.reshape(-1,1),gry.reshape(-1,1),gsy.reshape(-1,1),gJ.reshape(-1,1)
            # compute normals and surface Jacobian at Gauss points on face f1
            if(f1==0):
                gnx = -gsx;     gny = -gsy;   
            if(f1==1): 
                gnx =  grx+gsx; gny =  gry+gsy; 
            if(f1==2):
                gnx = -grx;     gny = -gry;     
            
            gsJ = num.sqrt(gnx*gnx+gny*gny);
            gnx = gnx/gsJ;  gny = gny/gsJ;  gsJ = gsJ*gJ;
            # store normals and coordinates at Gauss nodes
            temp_gnx[:,f1] = gnx.T  
            temp_gx[:,f1]  = num.dot(gV[:,:,f1],x1);
            temp_gny[:,f1] = gny.T;  temp_gy[:,f1]  = num.dot(gV[:,:,f1],y1);

            #store Vandermondes for '-' and '+' traces
            temp_gVM[:,:,f1] = gV[:,:,f1];
            temp_gVP[:,:,f1] = gV[::-1,:,f2];

            # compute and store matrix to lift Gauss node data
            temp = (gw*gsJ)
            
            temp_glift[:,:,f1] = num.dot(linalg.inv(cMM),num.dot(gV[:,:,f1].T,num.diag(temp[:,0])));

        #update them
        cinfo.gnx = temp_gnx
        cinfo.gny = temp_gny
        cinfo.gx  = temp_gx
        cinfo.gy  = temp_gy
        cinfo.gVM = temp_gVM
        cinfo.gVP = temp_gVP
        cinfo.glift = temp_glift
        
        test.append(cp.copy(cinfo))    
    return test

def Cubature2D(Corder):
    """function [cubR,cubS,cubW, Ncub] = Cubature2D(Corder)
    Purpose: provide multidimensional quadrature (i.e. cubature) rules to integrate up to Corder polynomials"""
     
    cub2D = io.loadmat('cub2D.mat')

    if(Corder<=28):
        cubR = cub2D['cub2D'][0][Corder-1][:,0];
        cubS = cub2D['cub2D'][0][Corder-1][:,1];
        cubW = cub2D['cub2D'][0][Corder-1][:,2] 
    else:
        cubNA = num.ceil( (Corder+1)/2);
        cubA,cubWA = JacobiGQ(0,0, cubNA-1);
        cubNB = num.ceil( (Corder+1)/2);
        cubB,cubWB = JacobiGQ(1,0, cubNB-1);
        
        cubA = num.outer(num.ones((cubNB)),cubA);
        cubB = num.outer(cubB,num.ones((1,cubNA)));
        cubR = 0.5*(1+cubA)*(1-cubB)-1;
        cubS = cubB;
        cubW = 0.5*num.outer(cubWB,cubWA);
        
        cubR = cubR.reshape(cubR.shape[0]*cubR.shape[1]);
        cubS = cubS.reshape(cubS.shape[0]*cubS.shape[1]);
        cubW = cubW.reshape(cubW.shape[0]*cubW.shape[1])
    
    Ncub = len(cubW);
    return  cubR,cubS,cubW, Ncub

def InterpMatrix2D(N,V,rout, sout):
    """function [IM] = InterpMatrix2D(rout, sout)
    Purpose: Compute local elemental interpolation matrix"""
    #compute Vandermonde at (rout,sout)
    Vout = Vandermonde2D(N, rout, sout);
    invV = linalg.inv(V)
    #build interpolation matrix
    IM = num.dot(Vout,invV);
    return IM

def Write_Mat(filename,field):
    """output the result in .mat format for matlab"""

    file = open(filename, "w")   # Open file for writing
    io.savemat(file, field, appendmat = True, format='5', long_field_names = False)
    file.close()
    return None
 


