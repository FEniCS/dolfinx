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
// Last changed: 2013-03-02

#include <boost/shared_ptr.hpp>

#include <dolfin/common/Array.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/fem/Assembler.h>
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

using namespace dolfin;

//-----------------------------------------------------------------------------
void HarmonicSmoothing::move(Mesh& mesh, const BoundaryMesh& new_boundary,
                             const std::string mode)
{
  // Now this works regardless of reorder_dofs_serial value
  const bool reorder_dofs_serial = parameters["reorder_dofs_serial"];
  if (!reorder_dofs_serial)
    warning("The function HarmonicSmoothing::move no longer needs "
            "parameters[\"reorder_dofs_serial\"] = false");

  const std::size_t D = mesh.topology().dim();
  const std::size_t d = mesh.geometry().dim();

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

  // Number of vertices
  const std::size_t num_vertices = mesh.num_vertices();

  // Mapping of dofs to mesh vertex numbers (excluding ghost dofs)
  const std::vector<std::size_t> vertex_to_dof_map =
                                   V->dofmap()->vertex_to_dof_map(mesh);

  // Number of dofs (excluding ghost dofs)
  const std::size_t num_dofs = vertex_to_dof_map.size();

  // Number of boundary dofs (excluding ghost dofs), set below
  std::size_t num_boundary_dofs = 0;

  // Mapping of new_boundary vertex numbers to mesh vertex numbers
  const MeshFunction<std::size_t>& vertex_map_mesh_func =
                                     new_boundary.entity_map(0);
  const std::size_t num_boundary_vertices = vertex_map_mesh_func.size();
  const std::vector<std::size_t> vertex_map(vertex_map_mesh_func.values(),
                      vertex_map_mesh_func.values() + num_boundary_vertices);

  // Dummy for handling ghost dofs
  // FIXME: Is this safe? Are dof numbers always uints?
  const dolfin::la_index ghost = -1;

  // Inversion of vertex_to_dof_map. Ghost nodes take dof number = ghost
  std::vector<dolfin::la_index> dof_to_vertex_map(num_vertices, ghost);
  for (std::size_t i = 0; i < num_dofs; i++)
    dof_to_vertex_map[vertex_to_dof_map[i]] = i;

  // Local-to-global dof numbers offset
  const dolfin::la_index offset = A.local_range(0).first;

  // Create arrays for setting bcs.
  // Their indexing does not matter - same ordering does.
  std::vector<dolfin::la_index> boundary_dofs;
  boundary_dofs.reserve(num_boundary_vertices);
  std::vector<std::size_t> boundary_vertices;
  boundary_vertices.reserve(num_boundary_vertices);
  // TODO: We could use VertexIterator here
  for (std::size_t vert = 0; vert < num_boundary_vertices; vert++)
  {
    const dolfin::la_index dof = dof_to_vertex_map[vertex_map[vert]];
    if (dof != ghost)
    {
      // Global dof numbers
      boundary_dofs.push_back(dof + offset);

      // new_boundary vertex indices
      boundary_vertices.push_back(vert);

      num_boundary_dofs++;
    }
  }

  // Modify matrix (insert 1 on diagonal)
  A.ident(num_boundary_dofs, boundary_dofs.data());
  A.apply("insert");

  // Arrays for storing dirichlet condition and solution
  std::vector<double> boundary_values(num_boundary_dofs);
  std::vector<double> new_coordinates;
  new_coordinates.reserve(d*num_vertices);

  // Pick amg as preconditioner if available
  const std::string prec(has_krylov_solver_preconditioner("amg")
                         ? "amg" : "default");

  // We will need Function::compute_vertex_values()
  Function u(V);
  boost::shared_ptr<GenericVector> x(u.vector());

  // RHS vector
  Vector b(*x);

  // Solve system for each dimension
  for (std::size_t dim = 0; dim < d; dim++)
  {
    if (mode == "coordinates")
    {
      if (dim > 0)
        b.zero();

      // Initialize solution for faster convergence
      std::vector<double> initial_values(num_dofs);
      for (std::size_t dof = 0; dof < num_dofs; dof++)
        initial_values[dof] = mesh.geometry().x(vertex_to_dof_map[dof], dim);
      x->set_local(initial_values);

      // Store bc into RHS and solution so that CG solver can be used
      for (std::size_t i = 0; i < num_boundary_dofs; i++)
        boundary_values[i] = new_boundary.geometry().x(boundary_vertices[i], dim);
      b.set(boundary_values.data(), num_boundary_dofs, boundary_dofs.data());
      b.apply("insert");
      x->set(boundary_values.data(), num_boundary_dofs, boundary_dofs.data());
      x->apply("insert");
    }
    else if (mode == "displacement")
    {
      if (dim > 0)
        b.zero();

      // Store bc into RHS and solution so that CG solver can be used
      for (std::size_t i = 0; i < num_boundary_dofs; i++)
        boundary_values[i] = new_boundary.geometry().x(boundary_vertices[i], dim)
                           - mesh.geometry().x(vertex_map[boundary_vertices[i]], dim);
      b.set(boundary_values.data(), num_boundary_dofs, boundary_dofs.data());
      b.apply("insert");
      *x = b;
    }
    else
      dolfin_error("HarmonicSmoothing.cpp",
                   "move mesh harmonically",
                   "unknown mode = %s",
                   mode.c_str());

    // Solve system
    solve(A, *x, b, "cg", prec);

    // Get new coordinates
    // FIXME: compute_vertex_values could be avoided if vertex_to_dof_map would
    //        be also defined on ghost dofs. Now we don't know where in *x are
    //        stored ghost-nodes values. Version of get_local which would supply
    //        ghost values would be also helpful.
    // Note that for diplacement mode new_coordinates are not actully
    // new coordinates but displacement.
    std::vector<double> _new_coordinates;
    u.compute_vertex_values(_new_coordinates);
    new_coordinates.insert(new_coordinates.end(),
                           _new_coordinates.begin(),
                           _new_coordinates.end());
  }

  // Modify mesh coordinates
  MeshGeometry& geometry = mesh.geometry();
  std::vector<double> coord(d);
  if (mode == "coordinates")
    for (std::size_t i = 0; i < num_vertices; i++)
    {
      for (std::size_t dim = 0; dim < d; dim++)
        coord[dim] = new_coordinates[dim*num_vertices + i];
      geometry.set(i, coord);
    }
  else if (mode == "displacement")
    for (std::size_t i = 0; i < num_vertices; i++)
    {
      for (std::size_t dim = 0; dim < d; dim++)
        coord[dim] = new_coordinates[dim*num_vertices + i] + geometry.x(i, dim);
      geometry.set(i, coord);
    }
}
//-----------------------------------------------------------------------------