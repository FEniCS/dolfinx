"""Unit tests for SubDomain"""

# Copyright (C) 2013 Johan Hake
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2013-06-24 
# Last changed: 2013-06-24 

import unittest
import numpy as np
from dolfin import *

class SubDomainTester(unittest.TestCase):

    def test_compiled_subdomains(self):
        def noDefaultValues():
            CompiledSubDomain("a")

        def wrongDefaultType():
            CompiledSubDomain("a", a="1")
                
        def wrongParameterNames():
            CompiledSubDomain("long", str=1.0)

        self.assertRaises(RuntimeError, noDefaultValues)
        self.assertRaises(TypeError, wrongDefaultType)
        self.assertRaises(RuntimeError, wrongParameterNames)

    def test_creation_and_marking(self):
        
        class Left(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < DOLFIN_EPS

        class Right(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] > 1.0 - DOLFIN_EPS
        
        subdomain_pairs = [(Left(), Right()),
                           (AutoSubDomain(\
                               lambda x, on_boundary: x[0] < DOLFIN_EPS),
                            AutoSubDomain(\
                                lambda x, on_boundary: x[0] > 1.0 - DOLFIN_EPS)),
                           (CompiledSubDomain("near(x[0], a)", a = 0.0),
                            CompiledSubDomain("near(x[0], a)", a = 1.0)),
                           (CompiledSubDomain("near(x[0], 0.0)"),
                            CompiledSubDomain("near(x[0], 1.0)"))]
        
        for ind, MeshClass in enumerate([UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]):
            dim = ind+1
            args = [10]*dim
            mesh = MeshClass(*args)
            
            mesh.init()

            for left, right in subdomain_pairs:

                for MeshFunc, f_dim in [(VertexFunction, 0),
                                        (FacetFunction, dim-1),
                                        (CellFunction, dim)]:
                    f = MeshFunc("size_t", mesh, 0)
                
                    left.mark(f, 1)
                    right.mark(f, 2)
                    
                    correct = {(1,0):1,
                               (1,0):1,
                               (1,1):0,
                               (2,0):11,
                               (2,1):10,
                               (2,2):0,
                               (3,0):121,
                               (3,2):200,
                               (3,3):0}

                    # Check that the number of marked entities are at least the
                    # correct number (it can be larger in parallel)
                    self.assertTrue(all(value >= correct[dim, f_dim]
                                        for value in [
                                            MPI.sum(float((f.array()==2).sum())),
                                            MPI.sum(float((f.array()==1).sum())),
                                            ]))
                    
                    
if __name__ == "__main__":
    unittest.main()
