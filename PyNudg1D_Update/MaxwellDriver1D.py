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

#  1D Maxwell equation for cativity problem p67 NDG


# Make all numpy available via shorter 'num' prefix
import numpy as num
import scipy.linalg as linalg
import scipy.io as io

import PyNudg as nudg

# Driver script for solving the 1D Maxwell's equations
# Polynomial order used for approximation 
N = 8
Nv,VX,K,EToV = nudg.MeshGen1D(-1.0,1.0,10)

#------------------------------------------------------------------
#                             Startup.m
# Initialize solver and construct grid and metric
f = open('Startup.py')
exec f

# Set up material parameters
eps1 = num.hstack((num.ones((1, K/2)), 2.0*num.ones((1, K/2))))
mu1  = num.ones((1, K))
epsilon = num.ones((Np, 1))*eps1; mu = num.ones((Np, 1))*mu1

# Set initial conditions
E = num.sin(num.pi*x)*(x<0); H = num.zeros((Np, K))

# Solve Problem
FinalTime = 1.0
E, H      = nudg.Maxwell1D(E, H, epsilon, mu, FinalTime, G)

# output the result in matlab format
field = {'E_field' : E, 'H_field':H}

filename = "out.mat"
nudg.Write_Mat(filename,field)

