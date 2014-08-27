#!/usr/bin/env py.test

# Copyright (C) 2013 Garth N. Wells
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
# First added:  2013-05-12
# Last changed:

import unittest
from dolfin import * 
import os

# create an output folder
filepath = os.path.join(os.path.dirname(__file__), 'output', '')
if not os.path.exists(filepath):
    os.mkdir(filepath)

def test_save_mesh1D():
    mesh = UnitIntervalMesh(16)
    file = File(filepath + "mesh1D.x3d")
    #self.assertRaises(RuntimeError, file << mesh)

def test_save_mesh2D():
    mesh = UnitSquareMesh(16, 16)
    file = File(filepath + "mesh2D.x3d")
    file << mesh

def test_save_mesh3D():
    mesh = UnitCubeMesh(16, 16, 16)
    file = File(filepath + "mesh3D.x3d")
    file << mesh

def test_save_cell_meshfunction2D():
    mesh = UnitSquareMesh(16, 16)
    mf = CellFunction("size_t", mesh, 12)
    file = File(filepath + "cell_mf2D.x3d")
    file << mf

def test_save_facet_meshfunction2D():
    mesh = UnitSquareMesh(16, 16)
    mf = FacetFunction("size_t", mesh, 12)
    file = File(filepath + "facet_mf2D.x3d")
    #with pytest.raises(RuntimeError):
    #    file << mf

def test_save_cell_meshfunctio22D():
    mesh = UnitCubeMesh(16, 16, 16)
    mf = CellFunction("size_t", mesh, 12)
    file = File(filepath + "cell_mf3D.x3d")
    file << mf

def test_save_facet_meshfunction3D():
    mesh = UnitCubeMesh(16, 16, 16)
    mf = FacetFunction("size_t", mesh, 12)
    file = File(filepath + "facet_mf3D.x3d")
    #with pytest.raises(RuntimeError):
    #    file << mf
