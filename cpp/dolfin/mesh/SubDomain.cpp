// Copyright (C) 2007-2008 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SubDomain.h"
#include "Facet.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "MeshIterator.h"
#include "Vertex.h"
#include <dolfin/common/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
SubDomain::SubDomain(const double map_tol) : map_tolerance(map_tol)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SubDomain::~SubDomain()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EigenArrayXb SubDomain::inside(const Eigen::Ref<const EigenRowArrayXXd> x,
                               bool on_boundary) const
{
  throw std::runtime_error(
      "SubDomain::inside function not implemented by user");
  return EigenArrayXb();
}
//-----------------------------------------------------------------------------
void SubDomain::map(const Eigen::Ref<const EigenArrayXd> x,
                    Eigen::Ref<EigenArrayXd> y) const
{
  throw std::runtime_error("Function map() not implemented by user. (Required "
                           "for periodic boundary conditions)");
}
//-----------------------------------------------------------------------------
template <typename T>
void SubDomain::apply_markers(std::map<std::size_t, std::size_t>& sub_domains,
                              std::size_t dim, T sub_domain, const Mesh& mesh,
                              bool check_midpoint) const
{
  // FIXME: This function can probably be folded into the above
  //        function operator[] in std::map and MeshFunction.

  LOG(INFO) << "Computing sub domain markers for sub domain " << sub_domain;

  const int gdim = mesh.geometry().dim();
  const int D = mesh.topology().dim();

  // Compute connectivities for boundary detection, if necessary
  if (dim < D)
  {
    mesh.create_entities(dim);
    if (dim != D - 1)
      mesh.create_connectivity(dim, D - 1);
    mesh.create_connectivity(D - 1, D);
  }

  // Speed up the computation by only checking each vertex once (or
  // twice if it is on the boundary for some but not all facets).
  std::vector<bool> boundary_visited(mesh.num_entities(0), false);
  std::vector<bool> interior_visited(mesh.num_entities(0), false);
  std::vector<bool> boundary_inside(mesh.num_entities(0));
  std::vector<bool> interior_inside(mesh.num_entities(0));

  // Always false when marking cells
  bool on_boundary = false;

  // Compute sub domain markers
  for (auto& entity : MeshRange<MeshEntity>(mesh, dim))
  {
    // Check if entity is on the boundary if entity is a facet
    if (dim == D - 1)
      on_boundary = (entity.num_global_entities(D) == 1);
    else if (dim < D - 1)
    {
      // Or, if entity is of topological dimension less than D - 1,
      // check if any connected facet is on the boundary
      on_boundary = false;
      for (std::size_t f(0); f < entity.num_entities(D - 1); ++f)
      {
        std::size_t facet_id = entity.entities(D - 1)[f];
        Facet facet(mesh, facet_id);
        if (facet.num_global_entities(D) == 1)
        {
          on_boundary = true;
          break;
        }
      }
    }

    // Select the visited-cache to use for this entity
    std::vector<bool>& is_visited
        = (on_boundary ? boundary_visited : interior_visited);
    std::vector<bool>& is_inside
        = (on_boundary ? boundary_inside : interior_inside);

    // Start by assuming all points are inside
    bool all_points_inside = true;

    // Check all incident vertices if dimension is > 0 (not a vertex)
    if (entity.dim() > 0)
    {
      for (auto& vertex : EntityRange<Vertex>(entity))
      {
        const std::int32_t vertex_index = vertex.index();
        if (!is_visited[vertex_index])
        {
          is_visited[vertex_index] = true;
          is_inside[vertex_index] = inside(vertex.x(), on_boundary)[0];
        }

        if (!is_inside[vertex_index])
        {
          all_points_inside = false;
          break;
        }
      }
    }

    // Check midpoint (works also in the case when we have a single vertex)
    if (all_points_inside && check_midpoint)
    {
      Eigen::Map<EigenRowArrayXd> x(entity.midpoint().data(), gdim);
      if (!inside(x, on_boundary)[0])
        all_points_inside = false;
    }

    // Mark entity with all vertices inside
    if (all_points_inside)
      sub_domains[entity.index()] = sub_domain;
  }
}
//-----------------------------------------------------------------------------
