// Copyright (C) 2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-08-11
// Last changed: 2008-09-13

#include <dolfin/common/Array.h>
#include <dolfin/fem/Assembler.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/solve.h>
#include <dolfin/la/Vector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include "Poisson1D.h"
#include "Poisson2D.h"
#include "Poisson3D.h"
#include "HarmonicSmoothing.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void HarmonicSmoothing::move(Mesh& mesh, const BoundaryMesh& new_boundary)
{
  // Choose form and function space
  FunctionSpace* V = 0;
  Form* form = 0;
  const uint D = mesh.topology().dim();
  const uint d = mesh.geometry().dim();
  switch (D)
  {
  case 1:
    V    = new Poisson1D::FunctionSpace(mesh);
    form = new Poisson1D::BilinearForm(*V, *V);
    break;
  case 2:
    V    = new Poisson2D::FunctionSpace(mesh);
    form = new Poisson2D::BilinearForm(*V, *V);
    break;
  case 3:
    V    = new Poisson3D::FunctionSpace(mesh);
    form = new Poisson3D::BilinearForm(*V, *V);
    break;
  default:
    error("Illegal mesh dimension %d for harmonic mesh smoothing.", D);
  };

  // Assemble matrix
  Matrix A;
  Assembler::assemble(A, *form);

  // Initialize vector
  const uint N = mesh.num_vertices();
  Vector b(N);

  // Get array of dofs for boundary vertices
  const MeshFunction<unsigned int>& vertex_map = new_boundary.vertex_map();
  const uint num_dofs = vertex_map.size();
  const uint* dofs = vertex_map.values();

  // Modify matrix (insert 1 on diagonal)
  A.ident(num_dofs, dofs);
  A.apply("insert");

  // Solve system for each dimension
  double* values = new double[num_dofs];
  Array<double> new_coordinates(d*N);
  Vector x;
  for (uint dim = 0; dim < d; dim++)
  {
    // Get boundary coordinates
    for (uint i = 0; i < new_boundary.num_vertices(); i++)
      values[i] = new_boundary.geometry().x(i, dim);

    // Modify right-hand side
    b.set(values, num_dofs, dofs);
    b.apply("insert");

    // Solve system
    solve(A, x, b, "gmres", "amg_hypre");

    // Get new coordinates
    Array<double> _new_coordinates(N, new_coordinates.data().get() + dim*N);
    x.get_local(_new_coordinates);
  }

  // Modify mesh coordinates
  MeshGeometry& geometry = mesh.geometry();
  for (uint dim = 0; dim < d; dim++)
    for (uint i = 0; i < N; i++)
      geometry.set(i, dim, new_coordinates[dim*N + i]);

  // Clean up
  delete V;
  delete form;
  delete [] values;
}
//-----------------------------------------------------------------------------
