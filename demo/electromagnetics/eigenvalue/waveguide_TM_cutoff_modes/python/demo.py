""" 
This demo demonstrates the calculation and visualization of a TM (Transverse 
Magnetic) cutoff mode of a rectangular waveguide.

For more information regarding waveguides see 

http://www.ee.bilkent.edu.tr/~microwave/programs/magnetic/rect/info.htm

See the pdf in the parent folder and the following reference

The Finite Element in Electromagnetics (2nd Ed)
Jianming Jin [7.2.1 - 7.2.2]

"""
__author__ = "Evan Lezar evanlezar@gmail.com"
__date__ = "2008-08-22"
__copyright__ = "Copyright (C) 2008 Evan Lezar"


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

# specify the waveguide width and height in metres
width = 1
height = 0.5

# specify the mode of interest. moi = 1 : dominant (TM_{11}) mode
moi = 1

# create the mesh using a Rectangle
nx = int(width/height)
if nx == 0:
    nx = 1
            
mesh = Rectangle(0, width, 0, height, nx, 1, 0)

# refine if desired
mesh.refine()

# define the finite element.  For vector electromagnetic problems Nedelec vector 
# elements are used.
# Specify the degree of the approximation
degree = 2
element = FiniteElement("Nedelec", "triangle", degree)

# define the test and trial functions
v = TestFunction(element)
u = TrialFunction(element)

# define the forms - gererates an generalized eigenproblem of the form 
# [S]{h} = k_o^2[T]{h}
# with the eigenvalues k_o^2 representing the square of the cutoff wavenumber 
# and the corresponding right-eigenvector giving the coefficients of the 
# discrete system used to obtain the approximate field anywhere in the domain   
 
a = dot(curl_t(v), curl_t(u))*dx
L = dot(v, u)*dx

# Assemble the system matrices
# stiffness (S) and mass matrices (T)
S = PETScMatrix()
T = PETScMatrix()
assemble(a, mesh, tensor=S)
assemble(L, mesh, tensor=T)

# now solve the eigen system
esolver = SLEPcEigenSolver()
esolver.set("eigenvalue spectrum", "smallest real")
esolver.solve(S, T)

# the result should have real eigenvalues but due to rounding errors, some of 
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
    #print "Eigenvalue " + str(i) + ": " + str(lr) + " + i" + str(lc)
    # ensure that the real part is large enough and that the complex part is zero
    if (lr > 1) and (lc == 0):
        print "Dominant mode found"
        dominant_mode_index = i
        break
        

if dominant_mode_index < 0:
    print "Dominant mode not found"
    
# now get the mode of interest
mode_of_interest_index = dominant_mode_index + moi - 1

h_e = PETScVector(S.size(1))
h_e_complex = PETScVector(S.size(1))

(k_o_squared, lc) = esolver.getEigenpair(h_e, h_e_complex, mode_of_interest_index)

print "Cutoff wavenumber squared: %f" % k_o_squared
if lc != 0: 
    print "WARNING:  Wavenumber is complex: %f +i%f" % (k_o_squared, lc) 

# now to visualize the magnetic field we need to calculate the field at a number 
# of points
# first define a discrete function using the eigenvector values as basis 
# function coefficients
# NOTE:  The coefficients need to be passed to the Function constructor as a 
#  dolfin Vector

# initialize the function
magnetic_field = Function(element, mesh, h_e)

# now specify the points where the field must be calculated and calculate
# number of points per unit length
n = 20 

# allocate numpy arrays for the magnetic field (H - x and y components) and the 
# position where the field is calculated 
H = N.zeros(((n+1)*(n+1), 2),dtype=N.float64)
XY = N.zeros(((n+1)*(n+1), 2),dtype=N.float64)
for i in range(n+1):
    for j in range(n+1):
        p_idx = i*(n+1) + j # the index of the point in the array
        XY[p_idx, 0] = float(width*i)/n
        XY[p_idx, 1] = float(height*j)/n
        
        # evaluate the magnetic field.  Result is stored in H[p_idx,:]
        magnetic_field.eval(H[p_idx,:], XY[p_idx,:])

# now plot the result
try:
    import pylab as P
    P.figure()
    P.quiver(XY[:,0],XY[:,1],H[:,0],H[:,1])
    
    
    P.xlabel('x')
    P.ylabel('y')
    
    P.axis('equal')
    print 'Plot saved as "TM_mode"'
    P.savefig('TM_mode')
except:
    print "pylab not found.  No image displayed"
