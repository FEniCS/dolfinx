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

import pytest
import numpy
from dolfin import *
import os
from dolfin_utils.test import fixture, cd_tempdir

def test_save_mesh1D(cd_tempdir):
    mesh = UnitIntervalMesh(16)
    file = File("mesh1D.x3d")
    #self.assertRaises(RuntimeError, file << mesh)

def test_save_mesh2D(cd_tempdir):
    mesh = UnitSquareMesh(16, 16)
    file = File("mesh2D.x3d")
    file << mesh

def test_save_mesh3D(cd_tempdir):
    mesh = UnitCubeMesh(16, 16, 16)
    file = File("mesh3D.x3d")
    file << mesh

def test_save_cell_meshfunction2D(cd_tempdir):
    mesh = UnitSquareMesh(16, 16)
    mf = CellFunction("size_t", mesh, 12)
    file = File("cell_mf2D.x3d")
    file << mf

def test_save_facet_meshfunction2D(cd_tempdir):
    mesh = UnitSquareMesh(16, 16)
    mf = FacetFunction("size_t", mesh, 12)
    file = File("facet_mf2D.x3d")
    #with pytest.raises(RuntimeError):
    #    file << mf


def test_save_cell_meshfunctio22D(cd_tempdir):
    mesh = UnitCubeMesh(16, 16, 16)
    mf = CellFunction("size_t", mesh, 12)
    file = File("cell_mf3D.x3d")
    file << mf


def test_save_facet_meshfunction3D(cd_tempdir):
    mesh = UnitCubeMesh(16, 16, 16)
    mf = FacetFunction("size_t", mesh, 12)
    file = File("facet_mf3D.x3d")
    #with pytest.raises(RuntimeError):
    #    file << mf


def test_mesh_str():
    mesh = UnitCubeMesh(2, 2, 2)
    str = X3DOM.str(mesh)
    mesh = UnitSquareMesh(5, 3)
    str = X3DOM.str(mesh)


def test_mesh_html():
    mesh = UnitCubeMesh(2, 2, 2)
    str = X3DOM.html(mesh)
    mesh = UnitSquareMesh(5, 3)
    str = X3DOM.html(mesh)
    # test IPython display hook:
    html = mesh._repr_html_()


def test_x3dom_parameters():
    p = X3DOMParameters()

    # Test Representation

    # Get colour
    c0 = p.get_diffuse_color()

    # Set colour
    c0 = [0.1, 0.2, 0.1]
    p.set_diffuse_color(c0)
    c1 = p.get_diffuse_color()
    assert numpy.array_equal(c0, c1)

    c0 = (0.1, 0.2, 0.1)
    p.set_diffuse_color(c0)
    c1 = p.get_diffuse_color()
    assert numpy.array_equal(c0, c1)

    c0 = numpy.array([0.1, 0.2, 0.1])
    p.set_diffuse_color(c0)
    c1 = p.get_diffuse_color()
    assert numpy.array_equal(c0, c1)

    # Test wrong length color sequences
    with pytest.raises(TypeError):
        c0 = [0.1, 0.2, 0.1, 0.2]
        p.set_diffuse_color(c0)

    with pytest.raises(TypeError):
        c0 = [0.1, 0.2]
        p.set_diffuse_color(c0)

    # Test invalid RGB value
    with pytest.raises(RuntimeError):
        c0 = [0.1, 0.2, 1.2]
        p.set_diffuse_color(c0)
