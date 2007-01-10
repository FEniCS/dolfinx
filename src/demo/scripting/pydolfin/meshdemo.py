from dolfin import *

mesh = Mesh("cow05b.xml.gz")

vertex = vertices(mesh)
while not vertex.end():
    print "vertex(" + str(vertex.index()) + "): ",
    print "x: " + str(vertex.point()[0]),
    print "y: " + str(vertex.point()[1]),
    print "z: " + str(vertex.point()[2])
    vertex.increment()

cell = cells(mesh)
while not cell.end():
    print "cell(" + str(cell.index()) + "): ",
    print "h: " + str(cell.diameter())
    cell.increment()
