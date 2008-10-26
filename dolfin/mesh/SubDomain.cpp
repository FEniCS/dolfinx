// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-24
// Last changed: 2007-12-12

#include <dolfin/log/log.h>
#include "MeshEntityIterator.h"
#include "Vertex.h"
#include "SubDomain.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SubDomain::SubDomain()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SubDomain::~SubDomain()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool SubDomain::inside(const double* x, bool on_boundary) const
{
  error("Unable to determine if point is inside subdomain, function inside() not implemented by user.");
  return false;
}
//-----------------------------------------------------------------------------
void SubDomain::map(const double* x, double* y) const
{
  error("Mapping between subdomains missing for periodic boundary conditions, function map() not implemented by user.");
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshFunction<uint>& sub_domains, uint sub_domain) const
{
  message(1, "Computing sub domain markers for sub domain %d.", sub_domain);

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
  
  // Always false when not marking facets
  bool on_boundary = false;

  // Compute sub domain markers
  for (MeshEntityIterator entity(mesh, dim); !entity.end(); ++entity)
  {
    // Check if entity is on the boundary if entity is a facet
    if (dim == D - 1)
      on_boundary = entity->numEntities(D) == 1;

    bool all_vertices_inside = true;
    // Dimension of facet > 0, check incident vertices
    if (entity->dim() > 0)
    {
      for (VertexIterator vertex(*entity); !vertex.end(); ++vertex)
      {
        simple_array<const double> x(mesh.geometry().dim(), vertex->x());
        if (!inside(x, on_boundary))
        {
          all_vertices_inside = false;
          break;
        }
      }
    }
    // Dimension of facet == 0, so just check the vertex itself
    else
    {
      simple_array<const double> x(mesh.geometry().dim(), mesh.geometry().x(entity->index()));
      if (!inside(x, on_boundary))
        all_vertices_inside = false;
    }

    // Mark entity with all vertices inside
    if (all_vertices_inside)
      sub_domains(*entity) = sub_domain;
  }
}
//-----------------------------------------------------------------------------
