// Copyright (C) 2008-2011 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-08-11
// Last changed: 2012-02-01

#include <boost/shared_ptr.hpp>

#include <dolfin/common/Array.h>
#include <dolfin/fem/Assembler.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/solve.h>
#include <dolfin/la/Vector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshFunction.h>
#include "Poisson1D.h"
#include "Poisson2D.h"
#include "Poisson3D.h"
#include "HarmonicSmoothing.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void HarmonicSmoothing::move(Mesh& mesh, const BoundaryMesh& new_boundary)
{
  error("The function HarmonicSmoothing::move is broken. See https://bugs.launchpad.net/dolfin/+bug/1047641.");

  not_working_in_parallel("ALE::move");

  const uint D = mesh.topology().dim();
  const uint d = mesh.geometry().dim();

  // Choose form and function space
  boost::shared_ptr<FunctionSpace> V;
  boost::shared_ptr<Form> form;
  switch (D)
  {
  case 1:
    V.reset(new Poisson1D::FunctionSpace(mesh));
    form.reset(new Poisson1D::BilinearForm(V, V));
    break;
  case 2:
    V.reset(new Poisson2D::FunctionSpace(mesh));
    form.reset(new Poisson2D::BilinearForm(V, V));
    break;
  case 3:
    V.reset(new Poisson3D::FunctionSpace(mesh));
    form.reset(new Poisson3D::BilinearForm(V, V));
    break;
  default:
    dolfin_error("HarmonicSmoothing.cpp",
                 "move mesh using harmonic smoothing",
                 "Illegal mesh dimension (%d)", D);
  }

  // Assemble matrix
  Matrix A;
  Assembler assembler;
  assembler.assemble(A, *form);

  // Initialize RHS vector
  const std::size_t N = mesh.num_vertices();
  Vector b(N);

  // Get array of dofs for boundary vertices
  const MeshFunction<std::size_t>& vertex_map = new_boundary.vertex_map();
  const std::size_t num_dofs = vertex_map.size();
  const std::vector<DolfinIndex> dofs(vertex_map.values(), vertex_map.values() + num_dofs);

  // Modify matrix (insert 1 on diagonal)

  A.ident(num_dofs, dofs.data());
  A.apply("insert");

  // Solve system for each dimension
  std::vector<double> values(num_dofs);
  std::vector<double> new_coordinates;
  Vector x;

  // Pick amg as preconditioner if available
  const std::string prec(has_krylov_solver_preconditioner("amg") ? "amg" : "default");

  for (uint dim = 0; dim < d; dim++)
  {
    // Get boundary coordinates
    for (uint i = 0; i < new_boundary.num_vertices(); i++)
      values[i] = new_boundary.geometry().x(i, dim);

    // Modify right-hand side
    b.set(&values[0], num_dofs, dofs.data());
    b.apply("insert");

    // Solve system
    solve(A, x, b, "gmres", prec);

    // Get new coordinates
    std::vector<double> _new_coordinates;
    x.get_local(_new_coordinates);
    new_coordinates.insert(new_coordinates.end(), _new_coordinates.begin(), _new_coordinates.end());
  }

  // Modify mesh coordinates
  MeshGeometry& geometry = mesh.geometry();
  std::vector<double> coord(d);
  for (uint i = 0; i < N; i++)
  {
    for (uint dim = 0; dim < d; dim++)
      coord[dim] = new_coordinates[dim*N + i];
    geometry.set(i, coord);
  }
}
//-----------------------------------------------------------------------------
