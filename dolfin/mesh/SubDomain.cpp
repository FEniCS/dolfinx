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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Niclas Jansson 2009.
//
// First added:  2007-04-24
// Last changed: 2011-08-31

#include <dolfin/common/Array.h>
#include <dolfin/common/RangedIndexSet.h>
#include <dolfin/log/log.h>
#include <dolfin/log/Progress.h>
#include "Mesh.h"
#include "MeshData.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"
#include "Vertex.h"
#include "MeshFunction.h"
#include "MeshValueCollection.h"
#include "SubDomain.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SubDomain::SubDomain(const double map_tol) : map_tolerance(map_tol),
                                             _geometric_dimension(0)
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
  dolfin_error("SubDomain.cpp",
               "check whether point is inside subdomain",
               "Function inside() not implemented by user");
  return false;
}
//-----------------------------------------------------------------------------
void SubDomain::map(const Array<double>& x, Array<double>& y) const
{
  dolfin_error("SubDomain.cpp",
               "map points within subdomain",
               "Function map() not implemented by user. (Required for periodic boundary conditions)");
}
//-----------------------------------------------------------------------------
void SubDomain::mark_cells(Mesh& mesh,
                           std::size_t sub_domain,
                           bool check_midpoint) const
{
  mark(mesh, mesh.topology().dim(), sub_domain, check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark_facets(Mesh& mesh,
                            std::size_t sub_domain,
                            bool check_midpoint) const
{
  mark(mesh, mesh.topology().dim() - 1, sub_domain, check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(Mesh& mesh,
                     std::size_t dim,
                     std::size_t sub_domain,
                     bool check_midpoint) const
{
  //dolfin_assert(mesh.domains().markers(dim));
  //dolfin_error("Not yet updated (SubDomain::mark) ");
  apply_markers(mesh.domains().markers(dim), dim, sub_domain, mesh,
                check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshFunction<std::size_t>& sub_domains,
                     std::size_t sub_domain,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, *sub_domains.mesh(), check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshFunction<int>& sub_domains,
                     int sub_domain,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, *sub_domains.mesh(), check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshFunction<double>& sub_domains,
                     double sub_domain,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, *sub_domains.mesh(), check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshFunction<bool>& sub_domains,
                     bool sub_domain,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, *sub_domains.mesh(), check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshValueCollection<std::size_t>& sub_domains,
                     std::size_t sub_domain,
                     const Mesh& mesh,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, mesh, check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshValueCollection<int>& sub_domains,
                     int sub_domain,
                     const Mesh& mesh,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, mesh, check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshValueCollection<double>& sub_domains,
                     double sub_domain,
                     const Mesh& mesh,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, mesh, check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshValueCollection<bool>& sub_domains,
                     bool sub_domain,
                     const Mesh& mesh,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, mesh, check_midpoint);
}
//-----------------------------------------------------------------------------
std::size_t SubDomain::geometric_dimension() const
{
  // Check that dim has been set
  if (_geometric_dimension == 0)
  {
    dolfin_error("SubDomain.cpp",
                 "get geometric dimension",
                 "Dimension of subdomain has not been specified");
  }

  return _geometric_dimension;
}
//-----------------------------------------------------------------------------
template<typename S, typename T>
void SubDomain::apply_markers(S& sub_domains,
                              T sub_domain,
                              const Mesh& mesh,
                              bool check_midpoint) const
{
  log(TRACE, "Computing sub domain markers for sub domain %d.", sub_domain);

  // Get the dimension of the entities we are marking
  const std::size_t dim = sub_domains.dim();

  // Compute facet - cell connectivity if necessary
  const std::size_t D = mesh.topology().dim();
  if (dim == D - 1)
  {
    mesh.init(D - 1);
    mesh.init(D - 1, D);
  }

  // Set geometric dimension (needed for SWIG interface)
  _geometric_dimension = mesh.geometry().dim();

  // Speed up the computation by only checking each vertex once (or
  // twice if it is on the boundary for some but not all facets).
  RangedIndexSet boundary_visited(mesh.num_vertices());
  RangedIndexSet interior_visited(mesh.num_vertices());
  std::vector<bool> boundary_inside(mesh.num_vertices());
  std::vector<bool> interior_inside(mesh.num_vertices());

  // Always false when not marking facets
  bool on_boundary = false;

  // Compute sub domain markers
  Progress p("Computing sub domain markers", mesh.num_entities(dim));
  for (MeshEntityIterator entity(mesh, dim); !entity.end(); ++entity)
  {
    // Check if entity is on the boundary if entity is a facet
    if (dim == D - 1)
      on_boundary = (entity->num_global_entities(D) == 1);

    // Select the visited-cache to use for this facet (or entity)
    RangedIndexSet&    is_visited = (on_boundary ? boundary_visited : interior_visited);
    std::vector<bool>& is_inside  = (on_boundary ? boundary_inside  : interior_inside);

    // Start by assuming all points are inside
    bool all_points_inside = true;

    // Check all incident vertices if dimension is > 0 (not a vertex)
    if (entity->dim() > 0)
    {
      for (VertexIterator vertex(*entity); !vertex.end(); ++vertex)
      {
        if (is_visited.insert(vertex->index()))
        {
          Array<double> x(_geometric_dimension, const_cast<double*>(vertex->x()));
          is_inside[vertex->index()] = inside(x, on_boundary);
        }

        if (!is_inside[vertex->index()])
        {
          all_points_inside = false;
          break;
        }
      }
    }

    // Check midpoint (works also in the case when we have a single vertex)
    if (all_points_inside && check_midpoint)
    {
      Array<double> x(_geometric_dimension,
                      const_cast<double*>(entity->midpoint().coordinates()));
      if (!inside(x, on_boundary))
        all_points_inside = false;
    }

    // Mark entity with all vertices inside
    if (all_points_inside)
      sub_domains.set_value(entity->index(), sub_domain);

    p++;
  }
}
//-----------------------------------------------------------------------------
template<typename T>
void SubDomain::apply_markers(std::map<std::size_t, std::size_t>& sub_domains,
                              std::size_t dim,
                              T sub_domain,
                              const Mesh& mesh,
                              bool check_midpoint) const
{
  // FIXME: This function can probably be folded into the above
  //        function operator[] in std::map and MeshFunction.

  log(TRACE, "Computing sub domain markers for sub domain %d.", sub_domain);

  // Compute facet - cell connectivity if necessary
  const std::size_t D = mesh.topology().dim();
  if (dim == D - 1)
  {
    mesh.init(D - 1);
    mesh.init(D - 1, D);
  }

  // Set geometric dimension (needed for SWIG interface)
  _geometric_dimension = mesh.geometry().dim();

  // Speed up the computation by only checking each vertex once (or
  // twice if it is on the boundary for some but not all facets).
  RangedIndexSet boundary_visited(mesh.num_vertices());
  RangedIndexSet interior_visited(mesh.num_vertices());
  std::vector<bool> boundary_inside(mesh.num_vertices());
  std::vector<bool> interior_inside(mesh.num_vertices());

  // Always false when not marking facets
  bool on_boundary = false;

  // Compute sub domain markers
  Progress p("Computing sub domain markers", mesh.num_entities(dim));
  for (MeshEntityIterator entity(mesh, dim); !entity.end(); ++entity)
  {
    // Check if entity is on the boundary if entity is a facet
    if (dim == D - 1)
      on_boundary = (entity->num_global_entities(D) == 1);

    // Select the visited-cache to use for this facet (or entity)
    RangedIndexSet&    is_visited = (on_boundary ? boundary_visited : interior_visited);
    std::vector<bool>& is_inside  = (on_boundary ? boundary_inside  : interior_inside);

    // Start by assuming all points are inside
    bool all_points_inside = true;

    // Check all incident vertices if dimension is > 0 (not a vertex)
    if (entity->dim() > 0)
    {
      for (VertexIterator vertex(*entity); !vertex.end(); ++vertex)
      {
        if (is_visited.insert(vertex->index()))
        {
          Array<double> x(_geometric_dimension, const_cast<double*>(vertex->x()));
          is_inside[vertex->index()] = inside(x, on_boundary);
        }

        if (!is_inside[vertex->index()])
        {
          all_points_inside = false;
          break;
        }
      }
    }

    // Check midpoint (works also in the case when we have a single vertex)
    if (all_points_inside && check_midpoint)
    {
      Array<double> x(_geometric_dimension,
                      const_cast<double*>(entity->midpoint().coordinates()));
      if (!inside(x, on_boundary))
        all_points_inside = false;
    }

    // Mark entity with all vertices inside
    if (all_points_inside)
      sub_domains[entity->index()] =  sub_domain;

    p++;
  }
}
//-----------------------------------------------------------------------------
