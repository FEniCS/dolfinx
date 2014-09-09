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

import pytest
from dolfin import * 
import os
from dolfin_utils.test import fixture

# create an output folder
@fixture
def temppath():
    filedir = os.path.dirname(os.path.abspath(__file__))
    basename = os.path.basename(__file__).replace(".py", "_data")
    temppath = os.path.join(filedir, basename, "")
    if not os.path.exists(temppath):
        os.mkdir(temppath)
    return temppath

def test_save_mesh1D(temppath):
    mesh = UnitIntervalMesh(16)
    file = File(temppath + "mesh1D.x3d")
    #self.assertRaises(RuntimeError, file << mesh)

def test_save_mesh2D(temppath):
    mesh = UnitSquareMesh(16, 16)
    file = File(temppath + "mesh2D.x3d")
    file << mesh

def test_save_mesh3D(temppath):
    mesh = UnitCubeMesh(16, 16, 16)
    file = File(temppath + "mesh3D.x3d")
    file << mesh

def test_save_cell_meshfunction2D(temppath):
    mesh = UnitSquareMesh(16, 16)
    mf = CellFunction("size_t", mesh, 12)
    file = File(temppath + "cell_mf2D.x3d")
    file << mf

def test_save_facet_meshfunction2D(temppath):
    mesh = UnitSquareMesh(16, 16)
    mf = FacetFunction("size_t", mesh, 12)
    file = File(temppath + "facet_mf2D.x3d")
    #with pytest.raises(RuntimeError):
    #    file << mf

def test_save_cell_meshfunctio22D(temppath):
    mesh = UnitCubeMesh(16, 16, 16)
    mf = CellFunction("size_t", mesh, 12)
    file = File(temppath + "cell_mf3D.x3d")
    file << mf

def test_save_facet_meshfunction3D(temppath):
    mesh = UnitCubeMesh(16, 16, 16)
    mf = FacetFunction("size_t", mesh, 12)
    file = File(temppath + "facet_mf3D.x3d")
    #with pytest.raises(RuntimeError):
    #    file << mf
