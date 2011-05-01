"""This demo program demonstrates how to mark sub domains of a mesh
and store the sub domain markers as a mesh function to a DOLFIN XML
file.

The sub domain markers produced by this demo program are the ones used
for the Stokes demo programs.
"""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-15 -- 2011-01-25"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *

set_log_level(1)

# Sub domain for no-slip (mark whole boundary, inflow and outflow will overwrite)
class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Sub domain for inflow (right)
class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 1.0 - DOLFIN_EPS and on_boundary

# Sub domain for outflow (left)
class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS and on_boundary

# Read mesh
mesh = Mesh("dolfin-2.xml.gz")

# Create mesh functions over the cell facets
sub_domains = MeshFunction("uint", mesh, mesh.topology().dim() - 1)
sub_domains_bool = MeshFunction("bool", mesh, mesh.topology().dim() - 1)
sub_domains_double = MeshFunction("double", mesh, mesh.topology().dim() - 1)

# Mark all facets as sub domain 3
sub_domains.set_all(3)
sub_domains_bool.set_all(False)
sub_domains_double.set_all(0.3)

# Mark no-slip facets as sub domain 0, 0.0
noslip = Noslip()
noslip.mark(sub_domains, 0)
noslip.mark(sub_domains_double, 0.0)

# Mark inflow as sub domain 1, 01
inflow = Inflow()
inflow.mark(sub_domains, 1)
inflow.mark(sub_domains_double, 0.1)

# Mark outflow as sub domain 2, 0.2, True
outflow = Outflow()
outflow.mark(sub_domains, 2)
outflow.mark(sub_domains_double, 0.2)
outflow.mark(sub_domains_bool, True)

# Save sub domains to file
file = File("subdomains.xml")
file << sub_domains

# FIXME: Not implemented
#file_bool = File("subdomains_bool.xml")
#file_bool << sub_domains_bool

file_double = File("subdomains_double.xml")
file_double << sub_domains_double

# Save sub domains to VTK files
file = File("subdomains.pvd")
file << sub_domains

file = File("subdomains_double.pvd")
file << sub_domains_double


