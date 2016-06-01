// Copyright (C) 2008-2011 Solveig Bruvoll and Anders Logg
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
// Modified by Jan Blechta 2013
//
// First added:  2008-05-02
// Last changed: 2013-03-06

#include <vector>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/fem/fem_utils.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include "HarmonicSmoothing.h"
#include "ALE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::shared_ptr<MeshDisplacement> ALE::move(std::shared_ptr<Mesh> mesh,
                                            const BoundaryMesh& new_boundary)
{
  dolfin_assert(mesh);
  return HarmonicSmoothing::move(mesh, new_boundary);
}
//-----------------------------------------------------------------------------
std::shared_ptr<MeshDisplacement> ALE::move(std::shared_ptr<Mesh> mesh0,
                                            const Mesh& mesh1)
{
  dolfin_assert(mesh0);
  if (mesh0->geometry().degree() != 1 || mesh1.geometry().degree() != 1)
  {
    dolfin_error("ALE.cpp",
                 "move mesh",
                 "This function does not support higher-order mesh geometry");
  }

  // FIXME: Maybe this works in parallel but there is no obvious way
  //        to test it as SubMesh::init does not work in parallel
  not_working_in_parallel("Move coordinates of mesh0 according "
                          "to mesh1 with common global vertices");

  dolfin_assert(mesh0);

  // Extract boundary meshes
  BoundaryMesh boundary0(*mesh0, "exterior");
  BoundaryMesh boundary1(mesh1, "exterior");

  // Get vertex mappings
  dolfin_assert(mesh0->data().exists("parent_vertex_indices", 0));
  const std::vector<std::size_t>& local_to_global_0
    = mesh0->data().array("parent_vertex_indices", 0);

  dolfin_assert(mesh1.data().exists("parent_vertex_indices", 0));
  const std::vector<std::size_t>& local_to_global_1
    = mesh1.data().array("parent_vertex_indices", 0);

  const MeshFunction<std::size_t>& boundary_to_mesh_0 = boundary0.entity_map(0);
  const MeshFunction<std::size_t>& boundary_to_mesh_1 = boundary1.entity_map(0);

  // Build global-to-local vertex mapping for mesh
  std::map<std::size_t, std::size_t> global_to_local_0;
  for (std::size_t i = 0; i < local_to_global_0.size(); i++)
    global_to_local_0[local_to_global_0[i]] = i;

  // Build mapping from mesh vertices to boundary vertices
  std::map<std::size_t, std::size_t> mesh_to_boundary_0;
  for (std::size_t i = 0; i < boundary_to_mesh_0.size(); i++)
    mesh_to_boundary_0[boundary_to_mesh_0[i]] = i;

  // Iterate over vertices in boundary1
  for (VertexIterator v(boundary1); !v.end(); ++v)
  {
    // Get global vertex index (steps 1 and 2)
    const std::size_t global_vertex_index
      = local_to_global_1[boundary_to_mesh_1[v->index()]];

    // Get local vertex index for mesh0 if possible (step 3)
    std::map<std::size_t, std::size_t>::const_iterator it;
    it = global_to_local_0.find(global_vertex_index);
    if (it == global_to_local_0.end())
      continue;
    const std::size_t mesh_index_0 = it->second;

    // Get vertex index on boundary0 (step 4)
    it = mesh_to_boundary_0.find(mesh_index_0);
    if (it == mesh_to_boundary_0.end())
    {
      dolfin_error("ALE.cpp",
                   "move mesh using mesh smoothing",
                   "Non-matching vertex mappings");
    }
    const std::size_t boundary_index_0 = it->second;

    // Update vertex coordinate
    MeshGeometry& geom = boundary0.geometry();
    geom.set(boundary_index_0, v->x());
  }

  // Move mesh
  return HarmonicSmoothing::move(mesh0, boundary0);
}
//-----------------------------------------------------------------------------
void ALE::move(Mesh& mesh, const GenericFunction& displacement)
{
  if (mesh.geometry().degree() != 1)
  {
    dolfin_error("ALE.cpp",
                 "move mesh",
                 "This function does not support higher-order mesh geometry");
  }

  // Check dimensions
  const std::size_t gdim = mesh.geometry().dim();
  if (!((displacement.value_rank() == 0 && gdim == 1) ||
        (displacement.value_rank() == 1
         && gdim == displacement.value_dimension(0))))
  {
    dolfin_error("ALE.cpp",
                 "move mesh using mesh smoothing",
                 "Illegal value dimension of displacement function");
  }

  // Interpolate at vertices
  const std::size_t N = mesh.num_vertices();
  std::vector<double> vertex_values;
  displacement.compute_vertex_values(vertex_values, mesh);

  // Move vertex coordinates
  MeshGeometry& geometry = mesh.geometry();
  std::vector<double> x(gdim);
  for (std::size_t i = 0; i < N; i++)
  {
    for (std::size_t j = 0; j < gdim; j++)
      x[j] = geometry.x(i, j) + vertex_values[j*N + i];
    geometry.set(i, x.data());
  }
}
//-----------------------------------------------------------------------------
void ALE::move(Mesh& mesh, const Function& displacement)
{
  Function position(displacement.function_space());
  get_coordinates(position, mesh.geometry());
  *position.vector() += *displacement.vector();
  set_coordinates(mesh.geometry(), position);
}
//-----------------------------------------------------------------------------
