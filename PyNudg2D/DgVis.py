import pylab as pl


def Plot_mesh(EToV,VX,VY,filename):
    """visualize the mesh for the problem"""
    for t in EToV:
        t_ex = [t[0],t[1],t[2],t[0]]
        pl.plot(VX[t_ex],VY[t_ex],'r')
        #pl.text(VX[t_ex],VY[t_ex],'%d' % 4)

    pl.plot(VX,VY,'.',ms=10)
    pl.axis('equal')
    pl.xlabel('x')
    pl.ylabel('y')
    pl.title('triangluar mesh of ' + filename)

    pl.show()


