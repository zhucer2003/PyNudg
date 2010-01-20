import pylab as pl


def Plot_mesh(ax,EToV,VX,VY,filename):
    """visualize the mesh for the problem"""
    for t in EToV:
        t_ex = [t[0],t[1],t[2],t[0]]
        pl.plot(VX[t_ex],VY[t_ex],'r')

    ax.plot(VX,VY,'.',ms=10)

    return

def PlotMesh2D(ax,Nfp,Nfaces,K,x,y,Fx,Fy):#,filename):
    """function PlotMesh2D()"""
    #axis equal
    xmax = x.max(); xmin = x.min();
    ymax = y.max(); ymin = y.min();

    Lx = xmax-xmin;
    Ly = ymax-ymin;
    xmax = xmax+.05*Lx; xmin = xmin-.05*Lx;
    ymax = ymax+.05*Ly; ymin = ymin-.05*Ly;

    oFx = Fx.reshape(Nfp, Nfaces*K,order='F'); oFy = Fy.reshape(Nfp, Nfaces*K,order='F');

    ax.plot(oFx, oFy, 'b')
    ax.axis([xmin,xmax,ymin,ymax])
    pl.xlabel('x')
    pl.ylabel('y')

    return

