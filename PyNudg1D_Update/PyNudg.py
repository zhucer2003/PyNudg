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

#  1D module


from __future__ import division
from math import sqrt
import numpy as num
import scipy.linalg as linalg
import scipy.sparse as sparse

# 1D parameters
Nfaces = 2
Nfp = 1
NODETOL = 1e-10
eps = num.finfo(float).eps

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
    #pass
    def __init__(self,Np,Nv,VX,K,EToV,r,Dr,LIFT,x,rx,J,nx,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB):

        self.Np = Np
        self.Nv = Nv; self.VX   = VX; 
        self.K  = K ; self.EToV = EToV;
        self.r  = r ; #self.V    = V
        self.Dr = Dr; self.LIFT = LIFT;
        self.x  = x ; self.rx   = rx;
        self.J  = J ; self.nx   = nx; 
        self.Fscale  = Fscale; self.EToE = EToE;
        self.EToF  = EToF ; self.vmapM = vmapM;
        self.vmapP = vmapP; self.vmapB = vmapB
        self.mapB  = mapB 
                

    def setglobal(self):
        """function: G=Setglobal(G)
            Purpose:set up the global data"""
        Np = self.Np
        Nv = self.Nv; VX   = self.VX; 
        K  = self.K;  EToV = self.EToV;
        r  = self.r;  #V    = self.V
        Dr = self.Dr; LIFT = self.LIFT;
        x  = self.x;  rx   = self.rx;
        J  = self.J;  nx   = self.nx; 
        Fscale  = self.Fscale; EToE = self.EToE;
        EToF    = self.EToF;  vmapM = self.vmapM;
        vmapP   = self.vmapP; vmapB = self.vmapB
        mapB    = self.mapB 
        return Np,Nv,VX,K,EToV,r,Dr,LIFT,x,rx,J,nx,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB

def gamma(z):

    g = 1
    for i in range(1, num.int32(z)):
        g = g*i
       
    return g
	
def JacobiP(x,alpha,beta,N):

	""" function P = JacobiP(x,alpha,beta,N)
	     Purpose: Evaluate Jacobi Polynomial of type (alpha,beta) > -1
	              (alpha+beta <> -1) at points x for order N and
	              returns P[1:length(xp))]
	     Note   : They are normalized to be orthonormal."""
	N = num.int32(N)
	Nx = x.shape[0]
      
	# Storage for recursive construction
	PL = num.zeros((num.int32(Nx),num.int32(N+1)))

	# Initial values P_0(x) and P_1(x)
	gamma0 = num.power(2.,alpha+beta+1)/(alpha+beta+1.)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1)

	# 
	PL[:,0] = 1.0/sqrt(gamma0)
	if N==0:
            return PL[:,0]

	gamma1 = (alpha+1.)*(beta+1.)/(alpha+beta+3.)*gamma0
	PL[:,1] = ((alpha+beta+2.)*x/2. + (alpha-beta)/2.)/sqrt(gamma1)
	if N==1: 
		return PL[:,1]

	# Repeat value in recurrence.
	aold = 2./(2.+alpha+beta)*sqrt((alpha+1.)*(beta+1.)/(alpha+beta+3.))

	# Forward recurrence using the symmetry of the recurrence.
	for i in range(1, N):
		h1 = 2.*i+alpha+beta;

		foo = (i+1.)*(i+1.+alpha+beta)*(i+1.+alpha)*(i+1.+beta)/(h1+1.)/(h1+3.)
		anew = 2./(h1+2.)*sqrt(foo);

		bnew = -(alpha*alpha-beta*beta)/(h1*(h1+2.))
		PL[:,i+1] = ( -aold*PL[:,i-1] + num.multiply(x-bnew,PL[:,i]) )/anew
		aold =anew

	return PL[:,N]

def GradJacobiP(z, alpha, beta, N):

	""" function [dP] = GradJacobiP(z, alpha, beta, N);
	    Purpose: Evaluate the derivative of the orthonormal Jacobi
	 	   polynomial of type (alpha,beta)>-1, at points x
	           for order N and returns dP[1:length(xp))]"""

	Nx = num.int32(z.shape[0])
	dP = num.zeros((Nx, 1))
	if N==0:
	  dP[:] = 0.0
	else:	
	  dP = sqrt(N*(N+alpha+beta+1.))*JacobiP(z,alpha+1,beta+1, N-1)

	return dP


def Vandermonde1D(N,xp):

	""" function [V1D] = Vandermonde1D(N,xp)
	    Purpose : Initialize the 1D Vandermonde Matrix.
	 	    V_{ij} = phi_j(xp_i);"""
	
	Nx = num.int32(xp.shape[0])	
	N  = num.int32(N)
	V1D = num.zeros((Nx, N+1))
	
        for j in range(N+1):
		V1D[:,j] = JacobiP(xp, 0, 0, j).T # give the tranpose of Jacobi.p

        return V1D


def GradVandermonde1D(N,xp):


	""" function [DVr] = GradVandermonde1D(N,xp)
            Purpose : Initialize the gradient of the modal basis (i)
			at (r) at order N"""	

	Nx = num.int32(xp.shape[0])	
	N  = num.int32(N)
 
	DV1D = num.zeros((Nx, N+1))
	for j in range(0,N+1):
		DV1D[:,j] = GradJacobiP(xp, 0, 0, j).T

	return DV1D

def Dmatrix1D(N,r,V):

	""" function [Dr] = Dmatrix1D(N,r)
	    Purpose : Initialize the (r) differentiation matrices
                   on the interval,evaluated at (r) at order N"""

	Nx = num.int32(r.shape[0])	
	N  = num.int32(N)
 
        Vr = GradVandermonde1D(N,r)
        invV = linalg.inv(V)

        Dr = num.dot(Vr,invV)
        
	return Dr

def Lift1D(N,r,V):

        """ function [LIFT] = Lift1D()
            Purpose : Compute surface integral term in DG formulation """

        Np = num.int32(r.shape[0])
        Emat = num.zeros((Np,2))

        # Define Emat
        Emat[0,0]    = 1.0
        Emat[Np-1,1] = 1.0

        # inv(mass matrix)*\s_n (L_i,L_j)_{edge_n}
        LIFT = num.dot(V, num.dot(V.T,Emat))
        
        return LIFT

def GeometricFactors1D(x, Dr):

        """ function [rx,J] = GeometricFactors1D(x,Dr)
            Purpose  : Compute the metric elements for the local mappings of the 1D elements"""  

        xr = num.dot(Dr,x)  
        J  = xr 
        rx =1.0/J  # element-wise division 
        return rx, J

def  Normals1D(xp):

        """ function [nx] = Normals1D
            Purpose : Compute outward pointing normals at elements faces"""

        K  =  num.int32(xp.shape[0]-1) 
        nx = num.zeros((Nfp*Nfaces, K)); 

        # Define outward normals
        nx[0, :] = -1.0; nx[1, :] = 1.0;

        return nx

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

def  MeshGen1D(xmin,xmax,K):

        """
        Purpose  : Generate simple equidistant grid with K elements
        """
        Nv = K+1 
        # Generate node coordinates
        VX = num.zeros(Nv)
        for i in num.arange(Nv):
            VX[i] = (xmax-xmin)*(i-0.0)/(Nv-1.0) + xmin
     
        # read element to node connectivity
        EToV = num.zeros((K, 2))
        for k in num.arange(K):
            EToV[k,0] = k
            EToV[k,1] = k+1

        return Nv, VX, K, EToV

def  Connect1D(EToV):

        """ function [EToE, EToF] = Connect1D(EToV)
            Purpose  : Build global connectivity arrays for 1D grid based on standard 
                    need sparse matrix EToV input array from grid generator """
        Nfaces = 2
    
        #Find number of elements and vertices
        K = EToV.shape[0]
        TotalFaces = Nfaces*K
        Nv = K+1

        # List of local face to local vertex connections
        vn =num.array([0,1])

        # Build global face to node sparse array
        #SpFToV = spalloc(TotalFaces, Nv, 2*TotalFaces)
        SpFToV  = sparse.lil_matrix((TotalFaces,Nv))
   
        sk = 0
        for k in range(K):
            for face in range(Nfaces) :
                SpFToV[ sk, EToV[k, vn[face]]] = 1.0
                sk = sk+1
     
        # Build global face to global face sparse array
        speye = sparse.lil_matrix((TotalFaces,TotalFaces))
        speye.setdiag(num.ones(TotalFaces))
        SpFToF = SpFToV*SpFToV.T - speye
    

        # Find complete face to face connections
        faces1, faces2 = SpFToF.tocoo().row, SpFToF.tocoo().col
    
   
        # Convert face global number to element and face numbers
        element1 = num.floor( (faces1)/Nfaces ) 
        face1    = num.mod( (faces1), Nfaces ) 
        element2 = num.floor( (faces2)/Nfaces )  
        face2    = num.mod( (faces2), Nfaces ) 
        # Rearrange into Nelements x Nfaces sized arrays
        size = num.array([K,Nfaces])
        ind  = sub2ind(size, element1, face1)


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

        """ 
        function: IND = sub2ind(size,I,J) 
        Purpose:returns the linear index equivalent to the row and column subscripts I and J for a matrix of size
        siz. siz is a vector with ndim(A) elements (in this case, 2), where siz(1) is the number of rows and 
        siz(2) is the number of columns.
        """
        ind = I*size[1]+J
        return ind

def BuildMaps1D(Fmask,EToE,EToF,K,Np,x):

        """ function [vmapM, vmapP, vmapB, mapB] = BuildMaps1D
             Purpose: Connectivity and boundary tables for nodes given in the K # of elements,each with N+1 degrees of freedo                  volume nodes consecutively"""
        temp    = num.arange(K*Np)
        nodeids = temp.reshape(Np, K,order='F').copy()

        vmapM   = num.zeros((Nfp, Nfaces, K)) 
        vmapP   = num.zeros((Nfp, Nfaces, K)) 
        for k1 in range(K):
            for f1 in range(Nfaces):
            # find index of face nodes with respect to volume node ordering

                vmapM[:,f1,k1] = nodeids[Fmask[:,f1], k1]
      
         
        xtemp = x.reshape(K*Np,1,order='F').copy()

        for k1 in range(K):
            for f1 in range(Nfaces):
            # find neighbor
                k2 = EToE[k1,f1]; f2 = EToF[k1,f1]
    
                # find volume node numbers of left and right nodes 
                vidM = vmapM[:,f1,k1];vidP = vmapM[:,f2,k2]
                x1 = xtemp[num.int32(vidM)]; x2 = xtemp[num.int32(vidP)]

                # Compute distance matrix
                D = num.inner(x1 -x2,x1-x2 )
                if (D<NODETOL):
                    vmapP[:,f1,k1] = vidP   

        vmapP = vmapP.reshape(Nfp*Nfaces*K,1,order='F') 
        vmapM = vmapM.reshape(Nfp*Nfaces*K,1,order='F')

        # Create list of boundary nodes
        mapB = (vmapP==vmapM).nonzero() ; vmapB = vmapM[mapB]
        mapB = mapB[0]
        
        # Create specific left (inflow) and right (outflow) maps
        mapI = 1.0; mapO = K*Nfaces; vmapI = 1.0; vmapO = K*Np
        return num.int32(vmapM), num.int32(vmapP), num.int32(vmapB),num.int32( mapB)

def Maxwell1D(E,H,eps,mu,FinalTime,G):


        """function [E,H] = Maxwell1D(E,H,eps,mu,FinalTime)
            Purpose  : Integrate 1D Maxwell's until FinalTime starting with conditions (E(t=0),H(t=0)) and materials (eps,mu)"""
        # set up the parameters
        Np,Nv,VX,K,EToV,r,Dr,LIFT,x,rx,J,nx,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB =G.setglobal() # Setglobal(G)  

        time = 0

        # Runge-Kutta residual storage  
        resE = num.zeros((Np,K)); resH = num.zeros((Np,K)); 

        # compute time step size
        xmin = min(abs(x[0, :]-x[1, :]))
        CFL=1.0;  dt = CFL*xmin;
        Nsteps = num.ceil(FinalTime/dt); dt = FinalTime/float(Nsteps)

        # outer time step loop 
        for tstep in range(num.int32(Nsteps)):
            for INTRK in range(5):
                rhsE, rhsH = MaxwellRHS1D(E,H,eps,mu,G)     
                resE = rk4a[INTRK]*resE + dt*rhsE;
                resH = rk4a[INTRK]*resH + dt*rhsH;
      
                E = E+rk4b[INTRK]*resE;
                H = H+rk4b[INTRK]*resH; 
            # Increment time
            time = time+dt;

        return E,H

def  MaxwellRHS1D(E,H,eps,mu,G):

        """ function [rhsE, rhsH] = MaxwellRHS1D(E,H,eps,mu)
            Purpose  : Evaluate RHS flux in 1D Maxwell""" 
        # set up the parameters
        Np,Nv,VX,K,EToV,r,Dr,LIFT,x,rx,J,nx,Fscale,EToE,EToF,vmapM,vmapP,vmapB,mapB = G.setglobal() #Setglobal(G)  
 

        # Compute impedance
        Zimp = num.sqrt(mu/eps);

        # Define field differences at faces
        vmapM = vmapM.reshape(Nfp*Nfaces,K,order='F')
        vmapP = vmapP.reshape(Nfp*Nfaces,K,order='F')
        Im,Jm = ind2sub(vmapM,Np)
        Ip,Jp = ind2sub(vmapP,Np)
        dE = num.zeros(Nfp*Nfaces*K);
        dE = E[Im,Jm]-E[Ip,Jp];
        
        dH = num.zeros((Nfp*Nfaces,K)); 
        dH = H[Im,Jm]-H[Ip,Jp]; 
        Zimpm = num.zeros((Nfp*Nfaces,K)); 
        Zimpm = Zimp[Im,Jm]
        Zimpp = num.zeros((Nfp*Nfaces,K)); 
        Zimpp = Zimp[Ip,Jp]
        Yimpm = num.zeros((Nfp*Nfaces,K)); 
        Yimpm = 1/Zimpm
        Yimpp= num.zeros((Nfp*Nfaces,K)); 
        Yimpp = 1/Zimpp

        # Homogeneous boundary conditions, Ez=0
        size_H= Nfp*Nfaces
        I,J = ind2sub(mapB,size_H)
        Iz,Jz = ind2sub(vmapB,Np)

        Ebc = -E[Iz,Jz];  
        dE[I,J] = E[Iz,Jz] - Ebc; #why?
        Hbc =  H[Iz,Jz]; 
        dH[I,J] = H[Iz,Jz] - Hbc;
 

        # evaluate upwind fluxes
        fluxE = 1/(Zimpm + Zimpp)*(nx*Zimpp*dH - dE);
        fluxH = 1/(Yimpm + Yimpp)*(nx*Yimpp*dE - dH);

        # compute right hand sides of the PDE's
        rhsE = (-rx*num.dot(Dr,H) + num.dot(LIFT,Fscale*fluxE))/eps;
        rhsH = (-rx*num.dot(Dr,E) + num.dot(LIFT,Fscale*fluxH))/mu;

        return rhsE, rhsH

def ind2sub(matr,row_size):
    """function I,J = ind2sub(matr,row_size)
    purpose: convert linear index to 2D index""" 
    I = num.int32(num.mod(matr,row_size))
    J = num.int32((matr - I)/row_size)
    return I,J

def Write_Mat(filename,field):
    """output the result in .mat format for matlab"""

    file = open(filename, "w")   # Open file for writing
    io.savemat(file, field, appendmat = True, format='5', long_field_names = False)
    file.close()
    return None

