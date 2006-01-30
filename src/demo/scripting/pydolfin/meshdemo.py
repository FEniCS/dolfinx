import dolfin
mesh1 = dolfin.Mesh("cow05b.xml.gz")
vi = dolfin.VertexIterator(mesh1)
while not vi.end():
    vertex = vi.__deref__()
    print "vertex(" + str(vertex.id()) + "): ",
    print "x: " + str(vertex.coord().x),
    print "y: " + str(vertex.coord().y),
    print "z: " + str(vertex.coord().z)
    vi.increment()

ci = dolfin.CellIterator(mesh1)
while not ci.end():
    cell = ci.__deref__()
    print "cell(" + str(cell.id()) + "): ",
    print "h: " + str(cell.diameter())
    ci.increment()

