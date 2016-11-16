// Copyright (C) 2008-2011 Anders Logg, 2013 Jan Blechta
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
// Last changed: 2013-12-02

#include <dolfin/common/Array.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/fem/fem_utils.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/solve.h>
#include <dolfin/la/Vector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/function/Function.h>
#include "Poisson1D.h"
#include "Poisson2D.h"
#include "Poisson3D.h"
#include "HarmonicSmoothing.h"
#include <memory>

using namespace dolfin;

//-----------------------------------------------------------------------------
std::shared_ptr<MeshDisplacement>
HarmonicSmoothing::move(std::shared_ptr<Mesh> mesh,
                        const BoundaryMesh& new_boundary)
{
  dolfin_assert(mesh);
  if (mesh->geometry().degree() != 1 || new_boundary.geometry().degree() != 1)
  {
    dolfin_error("HarmonicSmoothing.cpp",
                 "move mesh using harmonic smoothing",
                 "This function does not support higher-order mesh geometry");
  }

  // Now this works regardless of reorder_dofs_serial value
  const bool reorder_dofs_serial = parameters["reorder_dofs_serial"];
  if (!reorder_dofs_serial)
  {
    warning("The function HarmonicSmoothing::move no longer needs "
            "parameters[\"reorder_dofs_serial\"] = false");
  }

  const std::size_t D = mesh->topology().dim();
  const std::size_t d = mesh->geometry().dim();

  // Choose form and function space
  std::shared_ptr<FunctionSpace> V;
  std::shared_ptr<Form> form;
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
  auto A = std::make_shared<Matrix>();
  assemble(*A, *form);

  // Number of mesh vertices (local)
  const std::size_t num_vertices = mesh->num_vertices();

  // Dof range
  const dolfin::la_index n0 = V->dofmap()->ownership_range().first;
  const dolfin::la_index n1 = V->dofmap()->ownership_range().second;
  const dolfin::la_index num_owned_dofs = n1 - n0;

  // Mapping of new_boundary vertex numbers to mesh vertex numbers
  const MeshFunction<std::size_t>& vertex_map_mesh_func
    = new_boundary.entity_map(0);
  const std::size_t num_boundary_vertices = vertex_map_mesh_func.size();
  const std::vector<std::size_t>
    vertex_map(vertex_map_mesh_func.values(),
               vertex_map_mesh_func.values() + num_boundary_vertices);

  // Mapping of mesh vertex numbers to dofs (including ghost dofs)
  const std::vector<dolfin::la_index> vertex_to_dofs = vertex_to_dof_map(*V);

  // Array of all dofs (including ghosts) with local numbering
  std::vector<dolfin::la_index> all_global_dofs(num_vertices);
  for (std::size_t i = 0; i < num_vertices; i++)
    all_global_dofs[i] = vertex_to_dofs[i];

  // Create arrays for setting bcs.  Their indexing does not matter -
  // same ordering does.
  std::size_t num_boundary_dofs = 0;
  std::vector<dolfin::la_index> boundary_dofs;
  boundary_dofs.reserve(num_boundary_vertices);
  std::vector<std::size_t> boundary_vertices;
  boundary_vertices.reserve(num_boundary_vertices);
  for (std::size_t vert = 0; vert < num_boundary_vertices; vert++)
  {
    // Skip ghosts
    const dolfin::la_index dof = vertex_to_dofs[vertex_map[vert]];
    if (dof < num_owned_dofs)
    {
      // Global dof numbers
      boundary_dofs.push_back(dof + n0);

      // new_boundary vertex indices
      boundary_vertices.push_back(vert);

      num_boundary_dofs++;
    }
  }

  // Modify matrix (insert 1 on diagonal)
  A->ident(num_boundary_dofs, boundary_dofs.data());
  A->apply("insert");

  // Arrays for storing Dirichlet condition and solution
  std::vector<double> boundary_values(num_boundary_dofs);
  std::vector<double> displacement;
  displacement.reserve(d*num_vertices);

  // Pick amg as preconditioner if available
  const std::string
    prec(has_krylov_solver_preconditioner("amg") ? "amg" : "default");

  // Displacement solution wrapped in Expression subclass
  // MeshDisplacement
  std::shared_ptr<MeshDisplacement> u(new MeshDisplacement(mesh));

  // RHS vector
  Vector b(*(*u)[0].vector());

  // Prepare solver
  // NOTE: GMRES needs to be used until Eigen a4b7b6e or 8dcc4ed is widespread;
  //       afterwards CG can be used again
  KrylovSolver solver(mesh->mpi_comm(), "bicgstab", prec);
  solver.parameters["nonzero_initial_guess"] = true;
  solver.set_operator(A);

  // Solve system for each dimension
  for (std::size_t dim = 0; dim < d; dim++)
  {
    // Get solution vector
    std::shared_ptr<GenericVector> x = (*u)[dim].vector();

    if (dim > 0)
      b.zero();

    // Store bc into RHS and solution so that CG solver can be used
    for (std::size_t i = 0; i < num_boundary_dofs; i++)
    {
      boundary_values[i] = new_boundary.geometry().x(boundary_vertices[i], dim)
        - mesh->geometry().x(vertex_map[boundary_vertices[i]], dim);
    }
    b.set(boundary_values.data(), num_boundary_dofs, boundary_dofs.data());
    b.apply("insert");
    *x = b;

    // Solve the system
    solver.solve(*x, b);

    // Get displacement
    std::vector<double> _displacement(num_vertices);
    x->get_local(_displacement.data(), num_vertices, all_global_dofs.data());
    displacement.insert(displacement.end(), _displacement.begin(),
                        _displacement.end());
  }

  // Modify mesh coordinates
  MeshGeometry& geometry = mesh->geometry();
  std::vector<double> coord(d);
  for (std::size_t i = 0; i < num_vertices; i++)
  {
    for (std::size_t dim = 0; dim < d; dim++)
      coord[dim] = displacement[dim*num_vertices + i] + geometry.x(i, dim);
    geometry.set(i, coord.data());
  }

  // Return calculated displacement
  return u;
}
//-----------------------------------------------------------------------------
