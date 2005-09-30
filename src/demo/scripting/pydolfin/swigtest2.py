import pydolfin
mesh1 = pydolfin.Mesh("cow05b.xml.gz")
ni = pydolfin.NodeIterator(mesh1)
node = ni.__deref__()
print node.coord().x
while not ni.end():
    node = ni.__deref__()
    print "node(" + str(node.id()) + "): ",
    print "x: " + str(node.coord().x),
    print "y: " + str(node.coord().y),
    print "z: " + str(node.coord().z)
    ni.increment()
