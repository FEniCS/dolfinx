// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2009.
//
// First added:  2007-04-24
// Last changed: 2011-01-25

#include <dolfin/common/Array.h>
#include <dolfin/log/log.h>
#include "Mesh.h"
#include "MeshData.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"
#include "SubDomain.h"
#include "Vertex.h"

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
  info(TRACE, "Computing sub domain markers for sub domain %d.", sub_domain);

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

  Array<double> x;

  // Compute sub domain markers
  Progress p("Computing sub domain markers", mesh.num_entities(dim));
  for (MeshEntityIterator entity(mesh, dim); !entity.end(); ++entity)
  {
    // Check if entity is on the boundary if entity is a facet
    if (dim == D - 1)
    {
      on_boundary = (entity->num_entities(D) == 1 &&
		     (!exterior || (((*exterior)[*entity]))));
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
