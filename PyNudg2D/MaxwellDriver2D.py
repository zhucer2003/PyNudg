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
import PyNudg2D as nudg

import time as ti# Driver script for solving the 2D vacuum Maxwell's equations on TM form

t1 = ti.time()
# Polynomial order used for approximation 
N = 5;

#Read in Mesh
Nv, VX, VY, K, EToV = nudg.MeshReaderGambit2D('Maxwell025.neu')

#Startup.m
# Initialize solver and construct grid and metric
f = open('Startup2D.py')
exec f

#set initial conditions
mmode = 1; nmode = 1;
Ez = num.sin(mmode*num.pi*x)*num.sin(nmode*num.pi*y); Hx = num.zeros((Np, K)); Hy = num.zeros((Np, K));

#Solve Problem
FinalTime = 1;
Hx,Hy,Ez,time = nudg.Maxwell2D(Hx,Hy,Ez,FinalTime, G);

t2 = ti.time()

print "time:", t2-t1
#output the result in matlab format
field = {'Ez_field' : Ez, 'Hx_field':Hx,\
         'Hy_field' : Hy, 'rx_field':rx,\
         'x_field' : x, 'y':y,\
         'time_field': time}

filename = "out.mat"

nudg.Write_Mat(filename,field)


