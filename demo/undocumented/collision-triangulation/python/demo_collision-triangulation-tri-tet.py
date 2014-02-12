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
# Last changed: 2014-02-10

import numpy
from dolfin import *
import sys

def get_triangle_mesh():
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh,2,3)
    editor.init_cells(2,1)
    editor.init_vertices(4,1)
    # add cells
    editor.add_cell(0,0,1,2)
    editor.add_cell(1,1,2,3)
    # add vertices
    editor.add_vertex(0,0,0,0)
    editor.add_vertex(1,1,0,0)
    editor.add_vertex(2,0,1,0)
    editor.add_vertex(3,1,1,0)
    editor.close()
    return mesh

def triangulation_to_mesh(triangulation):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh,2,3)
    num_cells = len(triangulation)/9
    num_vertices = len(triangulation)/3
    editor.init_cells(num_cells,1)
    editor.init_vertices(num_vertices,1)
    for i in xrange(num_cells):
        editor.add_cell(i, 3*i, 3*i+1, 3*i+2)
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


# compute the area of the intersection
def compute_area(triangulation):
    area = 0                      
    if (triangulation.size>0):
        tmesh = triangulation_to_mesh(triangulation)
        for t in cells(tmesh):
            if (t.volume()<0):
                print "negative area ", t.volume()
            area += t.volume()
    return area

# given the computed area and the exact area, return the error and
# relative error
def compute_errors(area,exactarea):
    error = area-exactarea
    if exactarea>0:
        relativeerror = (area-exactarea)/exactarea
    elif exactarea==0:
        relativeerror = error
    return (error,relativeerror)

# Test intersection calculation when mesh_B is placed in the vertices
# of mesh_A. In this test we avoid the case when the triangle mesh is
# inside and perfectly aligned with the tetrahedral mesh. For
# UnitCubeMesh(2,2,2) this happens for z=0.5.
def test_place_at_vertex():
    mesh_A = UnitCubeMesh(2,2,2)
    max_relativeerror = -1
    plotter = VTKPlotter(mesh_A)

    for xyz in mesh_A.coordinates():
        if (abs(xyz[2]-0.5)>1e-10):
            mesh_B = get_triangle_mesh()
            mesh_B.translate(Point(xyz[0],xyz[1],xyz[2]))
            exactarea = (1-abs(xyz[0]))*(1-abs(xyz[1]))
            
            triangulation = compute_intersection(mesh_A,mesh_B)
            area = compute_area(triangulation)
            [error,relativeerror] = compute_errors(area,exactarea)
            max_relativeerror = max(max_relativeerror,abs(relativeerror))
            
            # if (triangulation.size>0):
            #     plotter.plot(triangulation_to_mesh(triangulation))
            #     interactive()
            
            print "%f %f %f %1.6g %1.6g %1.6g" % (xyz[0],xyz[1],xyz[2],area,error,relativeerror)
            sys.stdout.flush()
    return max_relativeerror

# Tests intersection calculation when mesh_B is translated across
# mesh_A. The translation can be arbitrary as long as there is no
# rotation.
def test_translation(num_steps):
    x0 = -1
    y0 = -1
    z0 = 0
    mesh_A = UnitCubeMesh(2,2,2)
    max_relativeerror = -1
    plotter = VTKPlotter(mesh_A)

    for n in range(num_steps+1):
        x = x0+2*n/float(num_steps)
        y = y0+2*n/float(num_steps)

        for m in range(num_steps+1):
            z = z0+m/float(num_steps)
            
            # remove z==0.5
            if (abs(z-0.5)>1e-10):
                
                mesh_B = get_triangle_mesh()
                mesh_B.translate(Point(x,y,z))
                exactarea = (1-abs(x))*(1-abs(y))
                
                triangulation = compute_intersection(mesh_A,mesh_B)
                area = compute_area(triangulation)
                [error,relativeerror] = compute_errors(area,exactarea)
                max_relativeerror = max(max_relativeerror,abs(relativeerror))

                # if (triangulation.size>0):
                #     plotter.plot(triangulation_to_mesh(triangulation))
                #     interactive()

                print "%f %f %f %1.6g %1.6g %1.6g" % (x,y,z,area,error,relativeerror)
                sys.stdout.flush()
                
    return max_relativeerror

# Tests intersection calculation when mesh_B is rotated along the
# z-axis around the center (0.5,0.5,0.5). Note that the triangle mesh
# cannot be aligned with the tetrahedral mesh. If this is the case, we
# will get twice the area. This will happen for 0, 45 and 90 degrees.
def test_rotation(num_angles):
    angle_start = 0
    angle_step = 90./num_angles 
    mesh_A = UnitCubeMesh(2,2,2)
    max_relativeerror = -1
    plotter=VTKPlotter(mesh_A)

    for n in range(num_angles+1):
        angle = angle_start+n*angle_step
        if (abs(angle)>1e-10 and 
            abs(angle-45)>1e-10 and 
            abs(angle-90)>1e-10):
            # reload mesh_B since using mesh_B.rotate(angle_step)
            # gives poor accuracy
            mesh_B = get_triangle_mesh()
            mesh_B.translate(Point(0,0,0.5))
            mesh_B.rotate(angle,1)
            exactarea = 1
            
            triangulation = compute_intersection(mesh_A,mesh_B)
            area = compute_area(triangulation)
            [error,relativeerror] = compute_errors(area,exactarea)
            max_relativeerror = max(max_relativeerror,abs(relativeerror))
            
            # kind of fun to plot this:
            plotter.plot(triangulation_to_mesh(triangulation))
            
            print "%f %1.6g %1.6g %1.6g" % (angle,area,error,relativeerror)
            sys.stdout.flush()
    interactive()
    return max_relativeerror


# main
print "test place at vertex"
max_relativeerror_vertex = test_place_at_vertex()

print "test translation"
max_relativeerror_translation = test_translation(10)

print "test rotation"
max_relativeerror_rotation = test_rotation(90)

print "max relative error vertex ", max_relativeerror_vertex
print "max relative error translation ", max_relativeerror_translation
print "max relative error rotation ", max_relativeerror_rotation

