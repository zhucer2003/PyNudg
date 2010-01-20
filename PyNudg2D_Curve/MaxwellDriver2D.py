# PyNudg - the python Nodal DG Environment
# Copyright (C) 2009 xueyu zhu
#
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

# 2D vacuum Maxwell equation for cativity problem p201 NDG


# Make all numpy available via shorter 'num' prefix
import numpy as num
import scipy.linalg as linalg
import scipy.io as io
import scipy.special as special 
import PyNudg2D as nudg

import time as ti
# Driver script for solving the 2D vacuum Maxwell's equations on TM form
t1 = ti.time()

# Polynomial order used for approximation 
N = 7 

#Read in Mesh
Nv, VX, VY, K, EToV, BCType = nudg.MeshReaderGambitBC2D('circA01.neu')


##Startup.m
## Initialize solver and construct grid and metric
f = open('Startup2D.py')
exec f

#Set initial conditions
# First 6 modes of eigenmodes with 6 azimuthal periods
alpha = num.array([9.936109524217684,13.589290170541217,17.003819667816014,\
         20.320789213566506,23.586084435581391,26.820151983411403])

# choose radial mode
alpha0 = alpha[1]
theta = num.arctan2(y.real,x.real)
rad   = num.sqrt(x**2+y**2);

Ez = special.jn(6, alpha0*rad).real*num.cos(6*theta);
Hx = num.zeros((Np, K)); Hy = num.zeros((Np, K));

#Solve Problem for exactly one period
FinalTime = .1;
Hx,Hy,Ez,time = nudg.MaxwellCurved2D(Hx,Hy,Ez,FinalTime,G,V,curved);

# check the accuracy
exactEz = special.jn(6, alpha0*rad).real*num.cos(6*theta)*num.cos(alpha0*time);
maxabserror = (abs(Ez-exactEz)).max()

print "maxabserror =", maxabserror

t2 = ti.time()
print "time:", t2-t1

#output the result in matlab format
field = {'Ez_field' : Ez, 'Hx_field':Hx,\
         'Hy_field' : Hy, 'rx_field':rx,\
         'x_field'  : x,  'y_field ':y,\
         'time_field': time}

filename = "out.mat"

nudg.Write_Mat(filename,field)


