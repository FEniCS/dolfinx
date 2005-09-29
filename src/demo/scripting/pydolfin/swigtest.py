import pydolfin

mesh1 = pydolfin.Mesh("cow05b.xml.gz")
mesh2 = pydolfin.Mesh("cow05c.xml.gz")

print
print "mesh1 #nodes: " + str(mesh1.noNodes())
print "mesh2 #nodes: " + str(mesh2.noNodes())
print

print "Merging mesh2 with mesh1"
print

mesh1.merge(mesh2)

print
print "mesh1 #nodes: " + str(mesh1.noNodes())
print "mesh2 #nodes: " + str(mesh2.noNodes())
print

meshfile1 = pydolfin.File("mymesh.xml.gz")
meshfile1 << mesh1
