// Copyright (C) 2007-2008 Anders Logg
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
// Modified by Niclas Jansson 2009.
//
// First added:  2007-04-24
// Last changed: 2011-03-17

#include <dolfin/common/Array.h>
#include <dolfin/log/log.h>
#include "Mesh.h"
#include "MeshData.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"
#include "ParallelData.h"
#include "Vertex.h"
#include "SubDomain.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SubDomain::SubDomain() : _geometric_dimension(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SubDomain::~SubDomain()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool SubDomain::inside(const Array<double>& x, bool on_boundary) const
{
  error("Unable to determine if point is inside subdomain, function inside() not implemented by user.");
  return false;
}
//-----------------------------------------------------------------------------
void SubDomain::map(const Array<double>& x, Array<double>&) const
{
  error("Mapping between subdomains missing for periodic boundary conditions, function map() not implemented by user.");
}
//-----------------------------------------------------------------------------
/// Set sub domain markers (uint) for given subdomain
void SubDomain::mark(MeshFunction<uint>& sub_domains, uint sub_domain) const
{
  mark_meshfunction(sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshFunction<int>& sub_domains, int sub_domain) const
{
  mark_meshfunction(sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshFunction<double>& sub_domains, double sub_domain) const
{
  mark_meshfunction(sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshFunction<bool>& sub_domains, bool sub_domain) const
{
  mark_meshfunction(sub_domains, sub_domain);
}
//-----------------------------------------------------------------------------
dolfin::uint SubDomain::geometric_dimension() const
{
  // Check that dim has been set
  if (_geometric_dimension == 0)
    error("Internal error, dimension for subdomain has not been specified.");

  return _geometric_dimension;
}
//-----------------------------------------------------------------------------
template<class T>
void SubDomain::mark_meshfunction(MeshFunction<T>& sub_domains, T sub_domain) const
{
  log(TRACE, "Computing sub domain markers for sub domain %d.", sub_domain);

  // Get the dimension of the entities we are marking
  const uint dim = sub_domains.dim();

  // Compute facet - cell connectivity if necessary
  const Mesh& mesh = sub_domains.mesh();
  const uint D = mesh.topology().dim();
  if (dim == D - 1)
  {
    mesh.init(D - 1);
    mesh.init(D - 1, D);
  }

  // Set geometric dimension (needed for SWIG interface)
  _geometric_dimension = mesh.geometry().dim();

  // Always false when not marking facets
  bool on_boundary = false;

  // Extract exterior (non shared) facets markers
  const MeshFunction<bool>& exterior = mesh.parallel_data().exterior_facet();

  Array<double> x;

  // Compute sub domain markers
  Progress p("Computing sub domain markers", mesh.num_entities(dim));
  for (MeshEntityIterator entity(mesh, dim); !entity.end(); ++entity)
  {
    // Check if entity is on the boundary if entity is a facet
    if (dim == D - 1)
    {
      on_boundary = (entity->num_entities(D) == 1 && (exterior.size() == 0 || exterior[*entity]));
    }

    // Start by assuming all points are inside
    bool all_points_inside = true;

    // Check all incident vertices if dimension is > 0 (not a vertex)
    if (entity->dim() > 0)
    {
      for (VertexIterator vertex(*entity); !vertex.end(); ++vertex)
      {
        x.update(_geometric_dimension, const_cast<double*>(vertex->x()));
        if (!inside(x, on_boundary))
        {
          all_points_inside = false;
          break;
        }
      }
    }

    // Check midpoint (works also in the case when we have a single vertex)
    if (all_points_inside)
    {
      x.update(_geometric_dimension, const_cast<double*>(entity->midpoint().coordinates()));
      if (!inside(x, on_boundary))
        all_points_inside = false;
    }

    // Mark entity with all vertices inside
    if (all_points_inside)
      sub_domains[*entity] = sub_domain;

    p++;
  }
}
//-----------------------------------------------------------------------------
