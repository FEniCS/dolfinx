import dolfin
mesh1 = dolfin.Mesh("cow05b.xml.gz")
ni = dolfin.NodeIterator(mesh1)
while not ni.end():
    node = ni.__deref__()
    print "node(" + str(node.id()) + "): ",
    print "x: " + str(node.coord().x),
    print "y: " + str(node.coord().y),
    print "z: " + str(node.coord().z)
    ni.increment()

ci = dolfin.CellIterator(mesh1)
while not ci.end():
    cell = ci.__deref__()
    print "cell(" + str(cell.id()) + "): ",
    print "h: " + str(cell.diameter())
    ci.increment()

