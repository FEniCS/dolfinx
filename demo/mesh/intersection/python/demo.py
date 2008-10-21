# This demo shows the intersection of the boundary of a unit
# square (omega1) when a unit circle (omega2) is rotating
# around the unit square's origo.
#
# Original implementation: ../cpp/main.cpp by Kristoffer Selim
#

__author__ = "Kristoffer Selim (selim@simula.no)"
__date__ = "2008-10-14"
__copyright__ = "Copyright (C) 2008 Kristoffer Selim"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *
from numpy import *

# Create meshes
omega1 = UnitSquare(20,20);
omega2 = UnitCircle(80);


# Move and scale circle
coord = omega2.coordinates();
coord *= 0.5
coord += 1.0;

# Iterate over angle
theta = 0.0;
dtheta = 0.1*DOLFIN_PI;      
while theta < 2*DOLFIN_PI:
        
    # Compute intersection with boundary of square
    boundary = BoundaryMesh(omega1);        
    cells = ArrayUInt();
    omega2.intersection(boundary, cells, False);    
    
    # Create mesh function to plot intersection
    intersection = MeshFunction("uint", omega2, omega2.topology().dim());
    for j in range(cells.size()):
        intersection.set(cells[j],1);            
    
    # Plot intersection
    plot(intersection);
                    
    # Rotate circle around (0.5, 0.5)    
    x_r = coord[:,0] - 0.5; 
    y_r = coord[:,1] - 0.5;  
    coord[:,0] = 0.5 + (cos(dtheta)*x_r - sin(dtheta)*y_r);
    coord[:,1] = 0.5 + (sin(dtheta)*x_r + cos(dtheta)*y_r);
 
    theta += dtheta;



    

        
    


    


    
    

        
