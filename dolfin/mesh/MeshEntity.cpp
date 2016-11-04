// Copyright (C) 2006-2011 Anders Logg
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
// Modified by Andre Massing, 2009.
// Modified by Garth N. Wells, 2012.
//
// First added:  2006-05-11
// Last changed: 2016-11-04

#include <dolfin/log/log.h>
#include "Mesh.h"
#include "MeshTopology.h"
#include "Vertex.h"
#include "MeshEntity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshEntity::MeshEntity(const Mesh& mesh, std::size_t dim, std::size_t index)
  : _mesh(0), _dim(0), _local_index(0)
{
  init(mesh, dim, index);
}
//-----------------------------------------------------------------------------
void MeshEntity::init(const Mesh& mesh, std::size_t dim, std::size_t index)
{
  // Store variables
  _mesh = &mesh; // Yes, we should probably use a shared pointer here...
  _dim = dim;
  _local_index = index;

  // Check index range
  if (index < _mesh->num_entities(dim))
    return;

  // Initialize mesh entities
  _mesh->init(dim);

  // Check index range again
  if (index < _mesh->num_entities(dim))
    return;

  // Illegal index range
  dolfin_error("MeshEntity.cpp",
               "create mesh entity",
               "Mesh entity index %d out of range [0, %d] for entity of dimension %d",
               index, _mesh->num_entities(dim), dim);
}
//-----------------------------------------------------------------------------
MeshEntity::~MeshEntity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool MeshEntity::incident(const MeshEntity& entity) const
{
  // Must be in the same mesh to be incident
  if ( _mesh != entity._mesh )
    return false;

  // Get list of entities for given topological dimension
  const unsigned int* entities = _mesh->topology()(_dim, entity._dim)(_local_index);
  const std::size_t num_entities = _mesh->topology()(_dim, entity._dim).size(_local_index);

  // Check if any entity matches
  for (std::size_t i = 0; i < num_entities; ++i)
    if (entities[i] == entity._local_index)
      return true;

  // Entity was not found
  return false;
}
//-----------------------------------------------------------------------------
std::size_t MeshEntity::index(const MeshEntity& entity) const
{
  // Must be in the same mesh to be incident
  if (_mesh != entity._mesh)
  {
    dolfin_error("MeshEntity.cpp",
                 "compute index of mesh entity",
                 "Mesh entity is defined on a different mesh");
  }

  // Get list of entities for given topological dimension
  const unsigned int* entities = _mesh->topology()(_dim, entity._dim)(_local_index);
  const std::size_t num_entities = _mesh->topology()(_dim, entity._dim).size(_local_index);

  // Check if any entity matches
  for (std::size_t i = 0; i < num_entities; ++i)
    if (entities[i] == entity._local_index)
      return i;

  // Entity was not found
  dolfin_error("MeshEntity.cpp",
               "compute index of mesh entity",
               "Mesh entity was not found");

  return 0;
}
//-----------------------------------------------------------------------------
Point MeshEntity::midpoint() const
{
  // Special case: a vertex is its own midpoint (don't check neighbors)
  if (_dim == 0)
    return _mesh->geometry().point(_local_index);

  // Other wise iterate over incident vertices and compute average
  std::size_t num_vertices = 0;

  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  for (VertexIterator v(*this); !v.end(); ++v)
  {
    x += v->point().x();
    y += v->point().y();
    z += v->point().z();
    num_vertices++;
  }

  dolfin_assert(num_vertices > 0);

  x /= double(num_vertices);
  y /= double(num_vertices);
  z /= double(num_vertices);

  Point p(x, y, z);

  std::cout<< __FUNCTION__ << ' '; printf("%1.16f ",x); printf("%1.16f ",y); printf("%1.16f \n",z);

  //return p;

  // Test using Kahan summation
  auto Kaham_summation = [&](std::size_t d)
    {
      double sum = 0.0;
      double c = 0.0;
      for (VertexIterator v(*this); !v.end(); ++v)
      {
	const double y = v->point()[d];
	const double t = sum + y;
	c = (t - sum) - y;
	sum = t;
      }
      return sum;
    };

  const double xsum = Kaham_summation(0);
  const double ysum = Kaham_summation(1);
  const double zsum = Kaham_summation(2);
  const double xtmp = xsum / num_vertices;
  const double ytmp = ysum / num_vertices;
  const double ztmp = zsum / num_vertices;
  std::cout<< __FUNCTION__ << " Kahan "; printf("%1.16f ",xtmp); printf("%1.16f ",ytmp); printf("%1.16f \n",ztmp);

  Point q(xtmp,ytmp,ztmp);
  return q;

}
//-----------------------------------------------------------------------------
unsigned int MeshEntity::owner() const
{
  if (_dim != _mesh->topology().dim())
    dolfin_error("MeshEntity.cpp",
                 "get ownership of entity",
                 "Entity ownership is only defined for cells");

  const std::size_t offset = _mesh->topology().ghost_offset(_dim);
  if (_local_index < offset)
    dolfin_error("MeshEntity.cpp",
                 "get ownership of entity",
                 "Ownership of non-ghost cells is local process");

  return _mesh->topology().cell_owner()[_local_index - offset];
}
//-----------------------------------------------------------------------------
std::string MeshEntity::str(bool verbose) const
{
  if (verbose)
    warning("Verbose output for MeshEntityIterator not implemented.");

  std::stringstream s;
  s << "<Mesh entity " << index()
    << " of topological dimension " << dim() << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
