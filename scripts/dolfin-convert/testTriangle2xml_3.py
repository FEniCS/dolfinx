""" added by Jan Blechta 2012-11-23
This script test if conversion of test_Triangle_3.{edge, ele, mesh}
to test_Triangle_3.{xml, attr0.xml} by mesh-convert works fine. It particularly
checks if mean value of triangle attribute has right value on two domains.
"""

from dolfin import Mesh, MeshFunction, Expression, Constant, dx, assemble

mesh = Mesh('test_Triangle_3.xml')
mfun = MeshFunction('double', mesh, 'test_Triangle_3.attr0.xml')

class TopA(Expression):
    def eval_cell(self, value, x, ufc_cell):
        if x[1]>0:
            value[0] = mfun.array()[ufc_cell.index]
        else:
            value[0] = 0

class BottomA(Expression):
    def eval_cell(self, value, x, ufc_cell):
        if x[1]<0:
            value[0] = mfun.array()[ufc_cell.index]
        else:
            value[0] = 0

topA, bottomA = TopA(), BottomA()

volA    = assemble(Constant(0.5)*dx, mesh=mesh)
topA    = assemble(topA*dx,          mesh=mesh)
bottomA = assemble(bottomA*dx,       mesh=mesh)

print 'average of mesh function on top    A = ', topA   /volA, ', should be = -10.'
print 'average of mesh function on bottom A = ', bottomA/volA, ', should be =  10.'
