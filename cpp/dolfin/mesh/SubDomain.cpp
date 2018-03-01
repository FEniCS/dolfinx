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
#include "MeshValueCollection.h"
#include "Vertex.h"
#include <dolfin/common/RangedIndexSet.h>
#include <dolfin/log/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
SubDomain::SubDomain(const double map_tol)
    : map_tolerance(map_tol), _geometric_dimension(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SubDomain::~SubDomain()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Eigen::Matrix<bool, Eigen::Dynamic, 1>
SubDomain::inside(Eigen::Ref<const EigenRowMatrixXd> x, bool on_boundary) const
{
  log::dolfin_error("SubDomain.cpp", "check whether point is inside subdomain",
               "Function inside() not implemented by user");
  return Eigen::Matrix<bool, 0, 1>();
}
//-----------------------------------------------------------------------------
void SubDomain::map(Eigen::Ref<const Eigen::VectorXd> x,
                    Eigen::Ref<Eigen::VectorXd> y) const
{
  log::dolfin_error("SubDomain.cpp", "map points within subdomain",
               "Function map() not implemented by user. (Required for periodic "
               "boundary conditions)");
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshFunction<std::size_t>& sub_domains,
                     std::size_t sub_domain, bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, *sub_domains.mesh(), check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshFunction<int>& sub_domains, int sub_domain,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, *sub_domains.mesh(), check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshFunction<double>& sub_domains, double sub_domain,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, *sub_domains.mesh(), check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshFunction<bool>& sub_domains, bool sub_domain,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, *sub_domains.mesh(), check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshValueCollection<std::size_t>& sub_domains,
                     std::size_t sub_domain, const Mesh& mesh,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, mesh, check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshValueCollection<int>& sub_domains, int sub_domain,
                     const Mesh& mesh, bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, mesh, check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshValueCollection<double>& sub_domains,
                     double sub_domain, const Mesh& mesh,
                     bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, mesh, check_midpoint);
}
//-----------------------------------------------------------------------------
void SubDomain::mark(MeshValueCollection<bool>& sub_domains, bool sub_domain,
                     const Mesh& mesh, bool check_midpoint) const
{
  apply_markers(sub_domains, sub_domain, mesh, check_midpoint);
}
//-----------------------------------------------------------------------------
std::size_t SubDomain::geometric_dimension() const
{
  // Check that dim has been set
  if (_geometric_dimension == 0)
  {
    log::dolfin_error("SubDomain.cpp", "get geometric dimension",
                 "Dimension of subdomain has not been specified");
  }

  return _geometric_dimension;
}
//-----------------------------------------------------------------------------
template <typename S, typename T>
void SubDomain::apply_markers(S& sub_domains, T sub_domain, const Mesh& mesh,
                              bool check_midpoint) const
{
  log::log(TRACE, "Computing sub domain markers for sub domain %d.", sub_domain);

  // Get the dimension of the entities we are marking
  const std::size_t dim = sub_domains.dim();

  // Compute connectivities for boundary detection, if necessary
  const std::size_t D = mesh.topology().dim();
  if (dim < D)
  {
    mesh.init(dim);
    if (dim != D - 1)
      mesh.init(dim, D - 1);
    mesh.init(D - 1, D);
  }

  // Set geometric dimension (needed for SWIG interface)
  _geometric_dimension = mesh.geometry().dim();

  // Find all vertices on boundary
  // Set all to -1 (interior) to start with
  // If a vertex is on the boundary, give it an index from [0, count)
  std::vector<std::int32_t> boundary_vertex(mesh.num_entities(0), -1);
  std::size_t count = 0;
  for (auto& facet : MeshRange<Facet>(mesh))
  {
    if (facet.num_global_entities(D) == 1)
    {
      const std::int32_t* v = facet.entities(0);
      for (unsigned int i = 0; i != facet.num_entities(0); ++i)
        if (boundary_vertex[v[i]] == -1)
        {
          boundary_vertex[v[i]] = count;
          ++count;
        }
    }
  }

  // Check all vertices for "inside" with "on_boundary=false"
  Eigen::Map<const EigenRowMatrixXd> x(
      mesh.geometry().x().data(), mesh.num_entities(0), _geometric_dimension);
  EigenVectorXb all_inside = inside(x, false);
  dolfin_assert(all_inside.rows() == x.rows());

  // Check all boundary vertices for "inside" with "on_boundary=true"
  EigenRowMatrixXd x_bound(count, _geometric_dimension);
  for (std::int32_t i = 0; i != mesh.num_entities(0); ++i)
    if (boundary_vertex[i] != -1)
      x_bound.row(boundary_vertex[i]) = x.row(i);
  EigenVectorXb bound_inside = inside(x_bound, true);
  dolfin_assert(bound_inside.rows() == x_bound.rows());

  // Copy values back to vector, now -1="not on boundary anyway", 1="inside",
  // 0="not inside"
  for (std::int32_t i = 0; i != mesh.num_entities(0); ++i)
    if (boundary_vertex[i] != -1)
      boundary_vertex[i] = bound_inside(boundary_vertex[i]) ? 1 : 0;

  // Compute sub domain markers
  for (auto& entity : MeshRange<MeshEntity>(mesh, dim))
  {
    // An Entity is on_boundary if all its vertices are on the boundary
    bool on_boundary = true;
    // Assuming it is on boundary, also check if all points are "inside"
    bool all_points_inside = true;
    // Assuming it is not on boundary, check points in "all_inside" array
    bool all_points_inside_nobound = true;
    for (const auto& v : EntityRange<Vertex>(entity))
    {
      const auto& idx = v.index();
      on_boundary &= (boundary_vertex[idx] != -1);
      all_points_inside &= (boundary_vertex[idx] == 1);
      all_points_inside_nobound &= all_inside[idx];
    }

    // In the case of not being on the boundary, use other criterion
    if (!on_boundary)
      all_points_inside = all_points_inside_nobound;

    // Check midpoint (works also in the case when we have a single vertex)
    // FIXME: refactor for efficiency
    if (all_points_inside && check_midpoint)
    {
      Eigen::Map<Eigen::RowVectorXd> x(
          const_cast<double*>(entity.midpoint().coordinates()),
          _geometric_dimension);
      if (!inside(x, on_boundary)[0])
        all_points_inside = false;
    }

    // Mark entity with all vertices inside
    if (all_points_inside)
      sub_domains.set_value(entity.index(), sub_domain);
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void SubDomain::apply_markers(std::map<std::size_t, std::size_t>& sub_domains,
                              std::size_t dim, T sub_domain, const Mesh& mesh,
                              bool check_midpoint) const
{
  // FIXME: This function can probably be folded into the above
  //        function operator[] in std::map and MeshFunction.

  log::log(TRACE, "Computing sub domain markers for sub domain %d.", sub_domain);

  // Compute connectivities for boundary detection, if necessary
  const std::size_t D = mesh.topology().dim();
  if (dim < D)
  {
    mesh.init(dim);
    if (dim != D - 1)
      mesh.init(dim, D - 1);
    mesh.init(D - 1, D);
  }

  // Set geometric dimension (needed for SWIG interface)
  _geometric_dimension = mesh.geometry().dim();

  // Speed up the computation by only checking each vertex once (or
  // twice if it is on the boundary for some but not all facets).
  common::RangedIndexSet boundary_visited{{{0, mesh.num_vertices()}}};
  common::RangedIndexSet interior_visited{{{0, mesh.num_vertices()}}};
  std::vector<bool> boundary_inside(mesh.num_vertices());
  std::vector<bool> interior_inside(mesh.num_vertices());

  // Always false when marking cells
  bool on_boundary = false;

  // Compute sub domain markers
  for (auto& entity : MeshRange<MeshEntity>(mesh, dim))
  {
    // Check if entity is on the boundary if entity is a facet
    if (dim == D - 1)
      on_boundary = (entity.num_global_entities(D) == 1);
    // Or, if entity is of topological dimension less than D - 1, check if any
    // connected
    // facet is on the boundary
    else if (dim < D - 1)
    {
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
    common::RangedIndexSet& is_visited
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
        if (is_visited.insert(vertex.index()))
        {
          Eigen::Map<Eigen::RowVectorXd> x(const_cast<double*>(vertex.x()),
                                           _geometric_dimension);
          is_inside[vertex.index()] = inside(x, on_boundary)[0];
        }

        if (!is_inside[vertex.index()])
        {
          all_points_inside = false;
          break;
        }
      }
    }

    // Check midpoint (works also in the case when we have a single vertex)
    if (all_points_inside && check_midpoint)
    {
      Eigen::Map<Eigen::RowVectorXd> x(
          const_cast<double*>(entity.midpoint().coordinates()),
          _geometric_dimension);
      if (!inside(x, on_boundary)[0])
        all_points_inside = false;
    }

    // Mark entity with all vertices inside
    if (all_points_inside)
      sub_domains[entity.index()] = sub_domain;
  }
}
//-----------------------------------------------------------------------------
void SubDomain::set_property(std::string name, double value)
{
  log::dolfin_error("SubDomain.cpp", "set parameter",
               "This method should be overloaded in the derived class");
}
//-----------------------------------------------------------------------------
double SubDomain::get_property(std::string name) const
{
  log::dolfin_error("SubDomain.cpp", "get parameter",
               "This method should be overloaded in the derived class");
  return 0.0;
}
//-----------------------------------------------------------------------------
