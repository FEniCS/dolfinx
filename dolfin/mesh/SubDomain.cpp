// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2009.
//
// First added:  2007-04-24
// Last changed: 2010-02-11

#include <dolfin/log/log.h>
#include "MeshData.h"
#include "MeshEntityIterator.h"
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
  info(1, "Computing sub domain markers for sub domain %d.", sub_domain);

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
  MeshFunction<uint>* exterior = mesh.data().mesh_function("exterior facets");

  // Compute sub domain markers
  Progress p("Computing sub domain markers", mesh.num_entities(dim));
  for (MeshEntityIterator entity(mesh, dim); !entity.end(); ++entity)
  {
    // Check if entity is on the boundary if entity is a facet
    if (dim == D - 1)
      on_boundary = (entity->num_entities(D) == 1 &&
		     (!exterior || (((*exterior)[*entity]))));

    bool all_vertices_inside = true;
    // Dimension of facet > 0, check incident vertices
    if (entity->dim() > 0)
    {
      for (VertexIterator vertex(*entity); !vertex.end(); ++vertex)
      {
        if (!inside(vertex->x(), on_boundary))
        {
          all_vertices_inside = false;
          break;
       }
      }
    }
    // Dimension of facet == 0, so just check the vertex itself
    else
    {
      if (!inside(mesh.geometry().x(entity->index()), on_boundary))
        all_vertices_inside = false;
    }

    // Mark entity with all vertices inside
    if (all_vertices_inside)
      sub_domains[*entity] = sub_domain;

    p++;
  }
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
