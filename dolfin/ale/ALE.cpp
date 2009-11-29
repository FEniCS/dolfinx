// Copyright (C) 2008-2009 Solveig Bruvoll and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-02
// Last changed: 2009-10-08

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Vertex.h>
#include "TransfiniteInterpolation.h"
#include "HarmonicSmoothing.h"
#include "ALE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void ALE::move(Mesh& mesh, BoundaryMesh& new_boundary, ALEType method)
{
  switch (method)
  {
  case lagrange:
    cout << "Updating mesh coordinates using transfinite mean value interpolation (Lagrange)." << endl;
    TransfiniteInterpolation::move(mesh, new_boundary,
                                   TransfiniteInterpolation::interpolation_lagrange);
    break;
  case hermite:
    cout << "Updating mesh coordinates using transfinite mean value interpolation (Hermite)." << endl;
    TransfiniteInterpolation::move(mesh, new_boundary,
                                   TransfiniteInterpolation::interpolation_hermite);
    break;
  case harmonic:
    cout << "Updating mesh coordinates using harmonic smoothing." << endl;
    HarmonicSmoothing::move(mesh, new_boundary);
    break;
  default:
    error("Unknown method for ALE mesh smoothing.");
  }
}
//-----------------------------------------------------------------------------
void ALE::move(Mesh& mesh0, Mesh& mesh1, ALEType method)
{
  // Extract boundary meshes
  BoundaryMesh boundary0(mesh0);
  BoundaryMesh boundary1(mesh1);

  // Get vertex mappings
  MeshFunction<uint>* local_to_global_0  = mesh0.data().mesh_function("global vertex indices");
  MeshFunction<uint>* local_to_global_1  = mesh1.data().mesh_function("global vertex indices");
  MeshFunction<uint>* boundary_to_mesh_0 = boundary0.data().mesh_function("vertex map");
  MeshFunction<uint>* boundary_to_mesh_1 = boundary1.data().mesh_function("vertex map");
  assert(local_to_global_0);
  assert(local_to_global_1);
  assert(boundary_to_mesh_0);
  assert(boundary_to_mesh_1);

  // Build global-to-local vertex mapping for mesh
  std::map<uint, uint> global_to_local_0;
  for (uint i = 0; i < local_to_global_0->size(); i++)
    global_to_local_0[(*local_to_global_0)[i]] = i;

  // Build mapping from mesh vertices to boundary vertices
  std::map<uint, uint> mesh_to_boundary_0;
  for (uint i = 0; i < boundary_to_mesh_0->size(); i++)
    mesh_to_boundary_0[(*boundary_to_mesh_0)[i]] = i;

  // Iterate over vertices in boundary1
  const uint dim = mesh0.geometry().dim();
  for (VertexIterator v(boundary1); !v.end(); ++v)
  {
    // Get global vertex index (steps 1 and 2)
    const uint global_vertex_index = (*local_to_global_1)[(*boundary_to_mesh_1)[v->index()]];

    // Get local vertex index for mesh0 if possible (step 3)
    std::map<uint, uint>::const_iterator it;
    it = global_to_local_0.find(global_vertex_index);
    if (it == global_to_local_0.end())
      continue;
    const uint mesh_index_0 = it->second;

    // Get vertex index on boundary0 (step 4)
    it = mesh_to_boundary_0.find(mesh_index_0);
    if (it == mesh_to_boundary_0.end())
      error("Unable to move mesh, non-matching vertex mappings.");
    const uint boundary_index_0 = it->second;

    // Update vertex coordinate
    double* x = boundary0.geometry().x(boundary_index_0);
    for (uint i = 0; i < dim; i++)
      x[i] = v->x()[i];
  }

  // Move mesh
  mesh0.move(boundary0);
}
//-----------------------------------------------------------------------------
void ALE::move(Mesh& mesh, const Function& displacement)
{
  // Check dimensions
  const FiniteElement& element = displacement.function_space().element();
  const uint gdim = mesh.geometry().dim();
  if (!((element.value_rank() == 0 && gdim == 0) ||
        (element.value_rank() == 1 && gdim == element.value_dimension(0))))
    error("Unable to move mesh, illegal value dimension of displacement function.");

  // Interpolate at vertices
  const uint N = mesh.num_vertices();
  std::vector<double> vertex_values(N*gdim);
  displacement.compute_vertex_values(vertex_values, mesh);

  // Move vertex coordinates
  double* x = mesh.geometry().x();
  for (uint d = 0; d < gdim; d++)
  {
    for (uint i = 0; i < N; i++)
      x[i*gdim + d] += vertex_values[d*N + i];
  }
}
//-----------------------------------------------------------------------------
