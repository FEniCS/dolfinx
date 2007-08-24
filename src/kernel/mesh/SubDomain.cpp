// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-24
// Last changed: 2007-08-20

#include <dolfin/log.h>
#include <dolfin/MeshEntityIterator.h>
#include <dolfin/Vertex.h>
#include <dolfin/SubDomain.h>

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
bool SubDomain::inside(const real* x, bool on_boundary) const
{
  error("Unable to determine if point is inside subdomain, function inside() not implemented by user.");
  return false;
}
//-----------------------------------------------------------------------------
void SubDomain::map(const real* x, real* y) const
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
  Mesh& mesh = sub_domains.mesh();
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

    // Check if all vertices are inside
    bool all_vertices_inside = true;
    for (VertexIterator vertex(*entity); !vertex.end(); ++vertex)
    {
      simple_array<real> x(mesh.geometry().dim(), vertex->x());
      if (!inside(x, on_boundary))
      {
        all_vertices_inside = false;
        break;
      }
    }

    // Mark entity with all vertices inside
    if (all_vertices_inside)
      sub_domains(*entity) = sub_domain;
  }
}
//-----------------------------------------------------------------------------
