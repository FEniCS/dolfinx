""" This demo demonstrates the calculation and visualization of a TM
(Transverse Magnetic) cutoff mode of a rectangular waveguide.

For more information regarding waveguides see 

http://www.ee.bilkent.edu.tr/~microwave/programs/magnetic/rect/info.htm

See the pdf in the parent folder and the following reference

The Finite Element in Electromagnetics (2nd Ed)
Jianming Jin [7.2.1 - 7.2.2]

"""
__author__ = "Evan Lezar evanlezar@gmail.com"
__date__ = "2008-08-22"
__copyright__ = "Copyright (C) 2008 Evan Lezar"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *
import numpy as N

# Test for PETSc and SLEPc
try:
    dolfin.PETScMatrix
except:
    print "PyDOLFIN has not been configured with PETSc. Exiting."
    exit()
try:
    dolfin.SLEPcEigenSolver
except:
    print "PyDOLFIN has not been configured with SLEPc. Exiting."
    exit()

# Make sure we use the PETSc backend
dolfin_set("linear algebra backend", "PETSc")

# Specify the waveguide width and height in metres
width = 1.0
height = 0.5

# Create the mesh using a Rectangle
mesh = Rectangle(0, width, 0, height, 2, 1, 0)

# Refine if desired
# mesh.refine()

# Define the finite element
degree = 2
element = FiniteElement("Nedelec", "triangle", degree)

# Define the test and trial functions
v = TestFunction(element)
u = TrialFunction(element)

# Define the forms - gererates an generalized eigenproblem of the form 
# [S]{h} = k_o^2[T]{h}
# with the eigenvalues k_o^2 representing the square of the cutoff wavenumber 
# and the corresponding right-eigenvector giving the coefficients of the 
# discrete system used to obtain the approximate field anywhere in the domain   
s = dot(curl_t(v), curl_t(u))*dx
t = dot(v, u)*dx

# Assemble the system matrices stiffness (S) and mass matrices (T)
S = PETScMatrix()
T = PETScMatrix()
assemble(s, mesh, tensor=S)
assemble(t, mesh, tensor=T)

# Solve the eigensystem
esolver = SLEPcEigenSolver()
esolver.set("eigenvalue spectrum", "smallest real")
esolver.set("eigenvalue solver", "lapack");
esolver.solve(S, T)

# The result should have real eigenvalues but due to rounding errors, some of 
# the resultant eigenvalues may be small complex values. 
# only consider the real part

# Now, the system contains a number of zero eigenvalues (near zero due to 
# rounding) which are eigenvalues corresponding to the null-space of the curl 
# operator and are a mathematical construct and do not represent physically 
# realizable modes.  These are called spurious modes.  
# So, we need to identify the smallest, non-zero eigenvalue of the system - 
# which corresponds with cutoff wavenumber of the the dominant cutoff mode.
dominant_mode_index = -1
for i in range(S.size(1)):
    (lr, lc) = esolver.getEigenvalue(i)
    # print "Eigenvalue " + str(i) + ": " + str(lr) + " + i" + str(lc)
    # ensure that the real part is large enough and that the complex part is zero
    if (lr > 1) and (lc == 0):
        print "Dominant mode found"
        dominant_mode_index = i
        break
       

if dominant_mode_index >= 0:
    print ("Dominant mode found.  Cutoff Squared: %f\n" % lr)
    if ( lc != 0 ):
        print "WARNING: Cutoff mode is complex (%f + i%f)\n" % (lr, lc)
else:
    print "ERROR: Dominant mode not found"
