# Copyright (C) 2014 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2014-01-07
# Last changed: 2014-01-29

import numpy
from dolfin import *
import sys


# Specify the mesh (must be cube otherwise the volume calculations
# doesn't work)
def loadmesh():
    N = 2
    return UnitCubeMesh(N,N,N)

# Creating a mesh from a triangulation (for visualization and volume
# computation). Note that this function completely disregards common
# vertices and creates a completely disconnected mesh.
def triangulation_to_mesh_3d(triangulation):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh,3,3)
    num_cells = len(triangulation)/12
    num_vertices = len(triangulation)/3
    editor.init_cells(num_cells,1)
    editor.init_vertices(num_vertices,1)
    for i in xrange(num_cells):
        editor.add_cell(i, 4*i, 4*i+1, 4*i+2, 4*i+3)
    for i in xrange(num_vertices):
        editor.add_vertex(i, triangulation[3*i], triangulation[3*i+1], triangulation[3*i+2])
    editor.close()
    return mesh

# call the intersection calculation
def compute_intersection(mesh_A,mesh_B):
    triangulation = numpy.array([])
    for cellA in cells(mesh_A):
        for cellB in cells(mesh_B):
            if (cellA.collides(cellB)):
                T = cellA.triangulate_intersection(cellB)
                triangulation = numpy.append(triangulation,T)
    return triangulation

# compute the volume of the tetrahedra
def compute_volume(triangulation):
    volume = 0                      
    if (triangulation.size>0):
        tmesh = triangulation_to_mesh_3d(triangulation)
        for t in cells(tmesh):
            if (t.volume()<0):
                print "negative volume ", t.volume()
            volume += t.volume()
    return volume

# given the computed volume and the exact volume, return the error and
# relative error
def compute_errors(volume,exactvolume):
    error = volume-exactvolume
    if exactvolume>0:
        relativeerror = (volume-exactvolume)/exactvolume
    elif exactvolume==0:
        relativeerror = error
    return (error,relativeerror)



# Test intersection calculation when mesh_B is placed in the vertices
# of mesh_A
def test_place_at_vertex():
    mesh_A = loadmesh()
    max_relativeerror = -1
    for xyz in mesh_A.coordinates():
        mesh_B = loadmesh()
        mesh_B.translate(Point(xyz[0],xyz[1],xyz[2]))
        exactvolume = (1-abs(xyz[0]))*(1-abs(xyz[1]))*(1-abs(xyz[2]))
        
        triangulation = compute_intersection(mesh_A,mesh_B)
        volume = compute_volume(triangulation)
        [error,relativeerror] = compute_errors(volume,exactvolume)
        max_relativeerror = max(max_relativeerror,abs(relativeerror))
        
        # plot(triangulation_to_mesh_3d(triangulation))
        # interactive()

        print "%f %f %f %1.6g %1.6g %1.6g" % (xyz[0],xyz[1],xyz[2],volume,error,relativeerror)
        sys.stdout.flush()
    return max_relativeerror



# Tests intersection calculation when mesh_B is translated across
# mesh_A. The translation can be arbitrary as long as there is no
# rotation.
def test_translation(num_steps):
    x0 = -1
    y0 = -1
    z0 = -1
    mesh_A = loadmesh()
    plotter = VTKPlotter(mesh_A)
    max_relativeerror = -1
    for n in range(num_steps+1):
        x = x0+2*n/float(num_steps)
        y = y0+2*n/float(num_steps)
        z = z0+2*n/float(num_steps)
        mesh_B = loadmesh()
        mesh_B.translate(Point(x,y,z))

        exactvolume = (1-abs(x))*(1-abs(y))*(1-abs(z))
        triangulation = compute_intersection(mesh_A,mesh_B)

        # if (triangulation.size>0):
        #     plotter.plot(triangulation_to_mesh_3d(triangulation))
        #     interactive()

        volume = compute_volume(triangulation)
        [error,relativeerror] = compute_errors(volume,exactvolume)
        max_relativeerror = max(max_relativeerror,abs(relativeerror))
        print "%f %f %f %1.6g %1.6g %1.6g" % (x,y,z,volume,error,relativeerror)
        sys.stdout.flush()
    return max_relativeerror



# Tests intersection calculation when mesh_B is rotated along the
# z-axis around the center (0.5,0.5,0.5). We can only rotate angles
# [0,90] degrees.
def test_rotation(num_angles):
    angle_start = 0
    angle_step = 90./num_angles 
    mesh_A = loadmesh()    
    plotter = VTKPlotter(mesh_A)
    max_relativeerror = -1

    for n in range(num_angles+1):
        angle = angle_start+n*angle_step
        # reload mesh_B since using mesh_B.rotate(angle_step) gives
        # poor accuracy
        mesh_B = loadmesh()   
        mesh_B.rotate(angle,2)
        k=tan(angle*pi/180)
        if (k>0):
            # triangle area to remove is 4*b*h/2
            b=0.5-0.5/k*(sqrt(1+k*k)-1)
            h=0.5-0.5*(sqrt(1+k*k)-k)
            exactvolume=1-2*b*h;
        else:
            exactvolume=1
        triangulation = compute_intersection(mesh_A,mesh_B)
        volume = compute_volume(triangulation)
        [error,relativeerror] = compute_errors(volume,exactvolume)
        max_relativeerror = max(max_relativeerror,abs(relativeerror))
        
        # plotter.plot(triangulation_to_mesh_3d(triangulation))
        # interactive()
        
        print "%f %1.6g %1.6g %1.6g" % (angle,volume,error,relativeerror)
        sys.stdout.flush()
    return max_relativeerror



# main
print "test place at vertex"
max_relativeerror_vertex = test_place_at_vertex()

print "test translation"
max_relativeerror_translation = test_translation(10)

print "test rotation"
max_relativeerror_rotation = test_rotation(9)

print "max relative error vertex ", max_relativeerror_vertex
print "max relative error translation ", max_relativeerror_translation
print "max relative error rotation ", max_relativeerror_rotation

