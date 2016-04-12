# -*- coding: utf-8 -*-
"This module provides a simple SVG renderer of 2D and 1D meshes for use in ipython notebook."
from six.moves import xrange as range

# Copyright (C) 2013-2014 Martin Sandve Alnæs
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

def mesh2svg(mesh, display_width=800.0):
    "Ipython notebook svg rendering function for 1D and 2D meshes."
    c = mesh.cells()
    num_cells = c.shape[0]
    nv = c.shape[1]

    # TODO: Can we detect display_width from current ipython environment?
    # TODO: Extract boundary mesh only if num_cells is too large?
    # TODO: I think this should just work with a quadrilateral mesh but it's not tested
    # TODO: I think this should just work with 1D mesh embedded in 2D but it's not tested

    x = mesh.coordinates()
    num_vertices = x.shape[0]
    d = x.shape[1]
    if (d == 3):
        return None

    cellname = mesh.ufl_cell().cellname()
    assert d == 1 or d == 2
    assert cellname in ("interval", "triangle", "quadrilateral")

    # Compute mesh size
    strokewidth = 2
    mesh_origin = [x[:,k].min() for k in range(d)]
    x_min = [x[:,k].min() for k in range(d)]
    x_max = [x[:,k].max() for k in range(d)]
    if d == 1:
        mesh_width  = x_max[0]-x_min[0]
        mesh_height = 0
    elif d == 2:
        mesh_width  = x_max[0]-x_min[0]
        mesh_height = x_max[1]-x_min[1]

    # Compute display scaling
    scale = float(display_width / mesh_width)
    display_height = max(mesh_height * display_width / mesh_width, strokewidth)

    # Add padding to include vertex circles
    display_padding = 10*strokewidth
    display_width = 2*display_padding + display_width
    display_height = 2*display_padding + display_height

    # Build list of screen coordinate vertices
    vertices = []
    if d == 1:
        vertices = [(display_padding + int(scale*(x[i,0] - x_min[0])), display_padding)
                    for i in range(num_vertices)]
    elif d == 2:
        # Mirror y-axis because of svg coordinate system
        vertices = [(display_padding + int(scale*(x[i,0] - x_min[0])),
                     display_padding + int(scale*(x_max[1] - x[i,1])))
                    for i in range(num_vertices)]

    # Build list of edges
    if cellname == "interval":
        # Build list of unique edges in 1D case
        edges = [(c[i,0], c[i,1]) for i in range(num_cells)]
    elif cellname == "triangle": # Should in principle work for quadrilateral as well
        # Build list of unique edges in 2D case
        edges = set()
        for i in range(num_cells):
            for j in range(nv):
                e = (c[i,j], c[i,(j+1)%nv])
                edges.add(tuple(sorted(e)))
        edges = sorted(edges)
    else:
        edges = []
    # Build lines for all edges
    lines = [(vertices[e0], vertices[e1]) for e0,e1 in edges]

    # Render svg code
    radius = strokewidth

    svg_line = '<line x1="%%s" y1="%%s" x2="%%s" y2="%%s" style="fill:none; stroke:black; stroke-width:"%s" />' % (strokewidth,)
    svg_lines = "\n".join(svg_line % (line[0] + line[1]) for line in lines)

    svg_vertex = '<circle cx="%%d" cy="%%d" r="%d" stroke="black" stroke-width="0" fill="red" />' % (radius,)
    svg_vertices = "\n".join(svg_vertex % p for p in vertices)

    svg = '''
    <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="%d" height="%d">
    %s
    %s
    </svg>
    '''
    code = svg % (display_width, display_height, svg_lines, svg_vertices)
    return code
