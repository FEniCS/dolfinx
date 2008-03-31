# Demonstration of MeshFunction
#
# Original implementation: ../cpp/main.cpp by Ola Skavhaug.
#
__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-15 -- 2008-03-31"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

dolfin_set("debug level", 1)

# Read mesh from file
mesh = Mesh("../mesh2D.xml.gz")
 
# Read mesh function from file
file_in = File("../meshfunction.xml")
f = MeshFunction("real", mesh)
file_in >> f

# Write mesh function to file
out = File("meshfunction_out.xml");
out << f

# Plot mesh function
plot(f, interactive=True)
