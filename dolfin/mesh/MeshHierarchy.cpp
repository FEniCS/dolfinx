// Copyright (C) 2015 Chris Richardson
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

#include<map>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/refinement/refine.h>

#include "MeshHierarchy.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// void MeshHierarchy::refine(MeshHierarchy& refined_mesh_hierarchy,
//                            const MeshFunction<bool>& markers) const
// {
//   std::shared_ptr<Mesh> refined_mesh(new Mesh);

//   // Make sure markers are on correct mesh, i.e. finest of hierarchy
//   dolfin_assert(markers.mesh()->id() == _meshes.back()->id());
//   dolfin::refine(*refined_mesh, *_meshes.back(), markers);

//   refined_mesh_hierarchy._meshes = _meshes;
//   refined_mesh_hierarchy._meshes.push_back(refined_mesh);

//   refined_mesh_hierarchy._parent = std::make_shared<const MeshHierarchy>(*this);
// }
//-----------------------------------------------------------------------------
std::shared_ptr<const MeshHierarchy> MeshHierarchy::refine(
                           const MeshFunction<bool>& markers) const
{
  std::shared_ptr<Mesh> refined_mesh(new Mesh);
  std::shared_ptr<MeshHierarchy> refined_hierarchy(new MeshHierarchy);

  // Make sure markers are on correct mesh, i.e. finest of hierarchy
  dolfin_assert(markers.mesh()->id() == _meshes.back()->id());
  dolfin::refine(*refined_mesh, *_meshes.back(), markers);

  refined_hierarchy->_meshes = _meshes;
  refined_hierarchy->_meshes.push_back(refined_mesh);

  refined_hierarchy->_parent = std::make_shared<const MeshHierarchy>(*this);

  return refined_hierarchy;
}
//-----------------------------------------------------------------------------
void MeshHierarchy::impose_lock(MeshFunction<bool>& vmarkers, std::size_t index)
{
  auto m_it = vertex_lock.find(index);
  // If this is a 'locking' vertex, impose constraint
  // on vertices in m_it->second
  if (m_it != vertex_lock.end())
  {
    for (auto &r : m_it->second)
      if (vmarkers[r])
      {
        // Prevent removal of this vertex
        vmarkers[r] = false;
        // Propagate lock recursively
        impose_lock(vmarkers, r);
      }
  }
}
//-----------------------------------------------------------------------------
void MeshHierarchy::coarsen(const MeshFunction<bool>& markers)
{
  const Mesh& mesh = *(_meshes.back());

  // Make sure there is a parent Mesh
  dolfin_assert(_parent != NULL);
  // Make sure markers are on finest mesh
  dolfin_assert(markers.mesh()->id() == mesh.id());
  // Markers must be a CellFunction
  dolfin_assert(markers.dim() == mesh.topology().dim());

  // Mark vertices
  // FIXME: copy across process boundaries in parallel
  VertexFunction<bool> vmarkers(mesh, false);
  for (CellIterator c(mesh); !c.end(); ++c)
    if (markers[*c])
    {
      for (VertexIterator v(*c); !v.end(); ++v)
        vmarkers[*v] = true;
    }

  // Check for consistency rules, using vertex_lock
  // FIXME: in parallel
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    const std::size_t local_index = v->index();
    // Non-refining vertices impose constraints on other vertices
    // recursively
    if (vmarkers[local_index] == false)
      impose_lock(vmarkers, local_index);
  }

  // At this point, vmarkers should be such that
  // all vertices created on the finest mesh are marked correctly for
  // potential removal.

}
//-----------------------------------------------------------------------------
