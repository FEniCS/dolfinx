import dolfin

mesh = dolfin.Mesh("cow05b.xml.gz")

vertex = dolfin.VertexIterator(mesh)
while not vertex.end():
    print "vertex(" + str(vertex.id()) + "): ",
    print "x: " + str(vertex.coord().x),
    print "y: " + str(vertex.coord().y),
    print "z: " + str(vertex.coord().z)
    vertex.increment()

cell = dolfin.CellIterator(mesh)
while not cell.end():
    print "cell(" + str(cell) + "): ",
    print "h: " + str(cell.diameter())
    cell.increment()
