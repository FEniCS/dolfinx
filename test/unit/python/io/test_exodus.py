#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

"""Unit tests for the Exodus io library"""

# Copyright (C) 2013 Nico Schlömer
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
from dolfin import *
import os
from dolfin_utils.test import skip_if_not_exodus, fixture, cd_tempdir

@skip_if_not_exodus
def test_save_1d_mesh(cd_tempdir):
    """Test output of 1D Mesh to Exodus file"""
    mesh = UnitIntervalMesh(32)
    File("mesh.e") << mesh

@skip_if_not_exodus
def test_save_2d_mesh(cd_tempdir):
    """Test output of 2D Mesh to Exodus file"""
    mesh = UnitSquareMesh(32, 32)
    File("mesh.e") << mesh

@skip_if_not_exodus
def test_save_3d_mesh(cd_tempdir):
    """Test output of 3D Mesh to Exodus file"""
    mesh = UnitCubeMesh(8, 8, 8)
    File("mesh.e") << mesh

@skip_if_not_exodus
def test_save_1d_scalar(cd_tempdir):
    """Test output of 1D scalar Function to Exodus file"""
    mesh = UnitIntervalMesh(32)
    u = Function(FunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File("u.e") << u

@skip_if_not_exodus
def test_save_2d_scalar(cd_tempdir):
    """Test output of 2D scalar Function to Exodus file"""
    mesh = UnitSquareMesh(16, 16)
    u = Function(FunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File("u.e") << u

@skip_if_not_exodus
def test_save_3d_scalar(cd_tempdir):
    """Test output of 3D scalar Function to Exodus file"""
    mesh = UnitCubeMesh(8, 8, 8)
    u = Function(FunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File("u.e") << u

@skip_if_not_exodus
def test_save_2d_vector(cd_tempdir):
    """Test output of 2D vector Function to Exodus file"""
    mesh = UnitSquareMesh(16, 16)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File("u.e") << u

@skip_if_not_exodus
def test_save_3d_vector(cd_tempdir):
    """Test output of 3D vector Function to Exodus file"""
    mesh = UnitCubeMesh(8, 8, 8)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File("u.e") << u

@skip_if_not_exodus
def test_save_2d_tensor(cd_tempdir):
    """Test output of 2D tensor Function to Exodus file"""
    mesh = UnitSquareMesh(16, 16)
    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    u.vector()[:] = 1.0
    File("u.e") << u

#@skip_if_not_exodus
#def test_save_3d_tensor(cd_tempdir):
#    mesh = UnitCubeMesh(8, 8, 8)
#    u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
#    u.vector()[:] = 1.0
#    File("u.e") << u
