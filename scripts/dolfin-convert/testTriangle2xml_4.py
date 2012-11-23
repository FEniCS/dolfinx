""" added by Jan Blechta 2012-11-23
This script test if conversion of test_Triangle_3.{edge, ele, mesh}
to test_Triangle_3.{xml, attr0.xml} by mesh-convert works fine. It particularly
checks if edge marker has right value by integrating over these edges.
Note that marker of dim=1 is saved in XML mesh file and loaded automatically
with mesh.
"""

from dolfin import Mesh, Constant, ds, dS, assemble, edges, Edge

mesh = Mesh('test_Triangle_3.xml')

a = assemble(Constant(1)('-')*dS(0), mesh=mesh)
b = assemble(Constant(1)     *ds(0), mesh=mesh)
c = assemble(Constant(1)('-')*dS(1), mesh=mesh)
d = assemble(Constant(1)     *ds(1), mesh=mesh)

edges_length = 0
for e in edges(mesh):
    edges_length += e.length()

print a+d, ', should be = ', edges_length
print b,   ', should be = ', 0.
print c,   ', should be = ', 0.
