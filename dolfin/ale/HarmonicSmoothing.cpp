// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-11
// Last changed: 2008-09-13

#include <dolfin/log/log.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/solve.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/fem/assemble.h>
#include "Poisson1D.h"
#include "Poisson2D.h"
#include "Poisson3D.h"
#include "HarmonicSmoothing.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void HarmonicSmoothing::move(Mesh& mesh, Mesh& new_boundary)
{
  error("HarmonicSmoothing::move needs updating for new Function interface.");
/*
  // Choose form
  Form* form = 0;
  const uint D = mesh.topology().dim();
  const uint d = mesh.geometry().dim();
  switch (D)
  {
  case 1:
    form = new Poisson1DBilinearForm();
    break;
  case 2:
    form = new Poisson2DBilinearForm();
    break;
  case 3:
    form = new Poisson3DBilinearForm();
    break;
  default:
    error("Illegal mesh dimension %d for harmonic mesh smoothing.", D);
  };

  // Assemble matrix
  Matrix A;
  assemble(A, *form, mesh);

  // Initialize vector
  const uint N = mesh.numVertices();
  Vector b(N);

  // Get array of dofs for boundary vertices
  const MeshFunction<uint>* vertex_map = new_boundary.data().meshFunction("vertex map");
  dolfin_assert(vertex_map);
  const uint num_dofs = vertex_map->size();
  const uint* dofs = vertex_map->values();

  // Modify matrix (insert 1 on diagonal)
  A.ident(num_dofs, dofs);
  A.apply();

  // Solve system for each dimension
  double* values = new double[num_dofs];
  double* new_coordinates = new double[d*N];
  Vector x;
  for (uint dim = 0; dim < d; dim++)
  {
    // Get boundary coordinates
    for (uint i = 0; i < new_boundary.numVertices(); i++)
      values[i] = new_boundary.geometry().x(i, dim);
    
    // Modify right-hand side
    b.set(values, num_dofs, dofs);
    b.apply();
    
    // Solve system
    solve(A, x, b, gmres, amg_hypre);

    // Get new coordinates
    x.get(new_coordinates + dim*N);
  }

  // Modify mesh coordinates
  MeshGeometry& geometry = mesh.geometry();
  for (uint dim = 0; dim < d; dim++)
    for (uint i = 0; i < N; i++)
      geometry.set(i, dim, new_coordinates[dim*N + i]);

  // Clean up
  delete form;
  delete [] values;
  delete [] new_coordinates;
*/
}
//-----------------------------------------------------------------------------
