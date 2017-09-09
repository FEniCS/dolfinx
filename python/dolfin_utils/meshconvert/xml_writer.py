# -*- coding: utf-8 -*-
"""Module for formatting DOLFIN XML files."""

# Copyright (C) 2012 Garth N. Wells
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
# Modified by Jan Blechta, 2012.
#
# Last changed: 2012-11-22

from __future__ import print_function

# Write mesh header
def write_header_mesh(ofile, cell_type, dim):
    ofile.write("""\
<?xml version=\"1.0\" encoding=\"UTF-8\"?>

<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">
  <mesh celltype="%s" dim="%d">
""" % (cell_type, dim))

# Write graph header
def write_header_graph(ofile, graph_type):
    ofile.write("""\
<?xml version=\"1.0\" encoding=\"UTF-8\"?>

<dolfin xmlns:dolfin=\"http://www.fenicsproject.org/\">
  <graph type="%s">
""" % (graph_type))

# Write mesh footer
def write_footer_mesh(ofile):
    ofile.write("""\
  </mesh>
</dolfin>
""")

# Write graph footer
def write_footer_graph(ofile):
    ofile.write("""\
  </graph>
</dolfin>
""")

def write_header_vertices(ofile, num_vertices):
    "Write vertices header"
    print("Expecting %d vertices" % num_vertices)
    ofile.write("    <vertices size=\"%d\">\n" % num_vertices)

def write_footer_vertices(ofile):
    "Write vertices footer"
    ofile.write("    </vertices>\n")
    print("Found all vertices")

def write_header_edges(ofile, num_edges):
    "Write edges header"
    print("Expecting %d edges" % num_edges)
    ofile.write("    <edges size=\"%d\">\n" % num_edges)

def write_footer_edges(ofile):
    "Write edges footer"
    ofile.write("    </edges>\n")
    print("Found all edges")

def write_vertex(ofile, vertex, *args):
    "Write vertex"
    coords = " ".join(['%s="%.16e"' % (comp, num) for (comp, num) in zip(["x","y","z"], args)])
    ofile.write('      <vertex index="%d" %s/>\n' % \
                (vertex, coords))

def write_graph_vertex(ofile, vertex, num_edges, weight = 1):
    "Write graph vertex"
    ofile.write("      <vertex index=\"%d\" num_edges=\"%d\" weight=\"%d\"/>\n" % \
        (vertex, num_edges, weight))

def write_graph_edge(ofile, v1, v2, weight = 1):
    "Write graph edge"
    ofile.write("      <edge v1=\"%d\" v2=\"%d\" weight=\"%d\"/>\n" % \
        (v1, v2, weight))

def write_header_cells(ofile, num_cells):
    "Write cells header"
    ofile.write("    <cells size=\"%d\">\n" % num_cells)
    print("Expecting %d cells" % num_cells)

def write_footer_cells(ofile):
    "Write cells footer"
    ofile.write("    </cells>\n")
    print("Found all cells")

def write_cell_interval(ofile, cell, n0, n1):
    "Write cell (interval)"
    ofile.write("      <interval index=\"%d\" v0=\"%d\" v1=\"%d\"/>\n" % \
        (cell, n0, n1))

def write_cell_triangle(ofile, cell, n0, n1, n2):
    "Write cell (triangle)"
    ofile.write("      <triangle index=\"%d\" v0=\"%d\" v1=\"%d\" v2=\"%d\"/>\n" % \
        (cell, n0, n1, n2))

def write_cell_tetrahedron(ofile, cell, n0, n1, n2, n3):
    "Write cell (tetrahedron)"
    ofile.write("      <tetrahedron index=\"%d\" v0=\"%d\" v1=\"%d\" v2=\"%d\" v3=\"%d\"/>\n" % \
        (cell, n0, n1, n2, n3))

def write_header_domains(ofile):
    ofile.write("    <domains>\n")

def write_footer_domains(ofile):
    ofile.write("    </domains>\n")

def write_header_meshvaluecollection(ofile, name, dim, size, etype):
    ofile.write("    <mesh_value_collection name=\"%s\" type=\"%s\" dim=\"%d\" size=\"%d\">\n" % (name, etype, dim, size))

def write_entity_meshvaluecollection(ofile, dim, index, value, local_entity=0):
    ofile.write("      <value cell_index=\"%d\" local_entity=\"%d\" value=\"%d\"/>\n" % (index, local_entity, value))

def write_footer_meshvaluecollection(ofile):
    ofile.write("    </mesh_value_collection>\n")

def write_header_meshfunction(ofile, dimensions, size):
    header = """<?xml version="1.0" encoding="UTF-8"?>
<dolfin xmlns:dolfin="http://fenicsproject.org">
  <mesh_function type="uint" dim="%d" size="%d">
""" % (dimensions, size)
    ofile.write(header)

def write_header_meshfunction2(ofile):
    header = """<?xml version="1.0" encoding="UTF-8"?>
<dolfin xmlns:dolfin="http://fenicsproject.org">
  <mesh_function>
"""
    ofile.write(header)

def write_entity_meshfunction(ofile, index, value):
    ofile.write("""    <entity index=\"%d\" value=\"%d\"/>
""" % (index, value))

def write_footer_meshfunction(ofile):
    ofile.write("""  </mesh_function>
</dolfin>""")
