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
Eigen::Vector<bool, Eigen::Dynamic>
SubDomain::inside(Eigen::Ref<const EigenRowMatrixXd> x, bool on_boundary) const
{
  dolfin_error("SubDomain.cpp", "check whether point is inside subdomain",
               "Function inside() not implemented by user");
  return false;
}
//-----------------------------------------------------------------------------
void SubDomain::map(Eigen::Ref<const Eigen::VectorXd> x,
                    Eigen::Ref<Eigen::VectorXd> y) const
{
  dolfin_error("SubDomain.cpp", "map points within subdomain",
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
    dolfin_error("SubDomain.cpp", "get geometric dimension",
                 "Dimension of subdomain has not been specified");
  }

  return _geometric_dimension;
}
//-----------------------------------------------------------------------------
template <typename S, typename T>
void SubDomain::apply_markers(S& sub_domains, T sub_domain, const Mesh& mesh,
                              bool check_midpoint) const
{
  log(TRACE, "Computing sub domain markers for sub domain %d.", sub_domain);

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

  // Speed up the computation by only checking each vertex once (or
  // twice if it is on the boundary for some but not all facets).
  RangedIndexSet boundary_visited{{{0, mesh.num_vertices()}}};
  RangedIndexSet interior_visited{{{0, mesh.num_vertices()}}};
  std::vector<bool> boundary_inside(mesh.num_vertices());
  std::vector<bool> interior_inside(mesh.num_vertices());

  // Find out which vertices match the 'inside' condition
  const std::vector<double>& x = mesh.geometry().x();
  Eigen::Map<const Eigen::MatrixXd> vertex_coords(
      x.data(), x.size() / _geometric_dimension, _geometric_dimension);
  Eigen::Vector<bool, Eigen::Dynamic> vertex_inside = inside(vertex_coords);

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
      for (std::size_t f = 0; f < entity.num_entities(D - 1); ++f)
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
    RangedIndexSet& is_visited
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
        if (!vertex_inside[vertex.index()])
        {
          all_points_inside = false;
          break;
        }
      }
    }

    // Check midpoint (works also in the case when we have a single vertex)
    if (all_points_inside && check_midpoint)
    {
      Eigen::Map<Eigen::VectorXd> x(
          const_cast<double*>(entity.midpoint().coordinates()),
          _geometric_dimension);
      if (!inside(x, on_boundary))
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

  log(TRACE, "Computing sub domain markers for sub domain %d.", sub_domain);

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
  RangedIndexSet boundary_visited{{{0, mesh.num_vertices()}}};
  RangedIndexSet interior_visited{{{0, mesh.num_vertices()}}};
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
    RangedIndexSet& is_visited
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
          Eigen::Map<Eigen::VectorXd> x(const_cast<double*>(vertex.x()),
                                        _geometric_dimension);
          is_inside[vertex.index()] = inside(x, on_boundary);
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
      Eigen::Map<Eigen::VectorXd> x(
          const_cast<double*>(entity.midpoint().coordinates()),
          _geometric_dimension);
      if (!inside(x, on_boundary))
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
  dolfin_error("SubDomain.cpp", "set parameter",
               "This method should be overloaded in the derived class");
}
//-----------------------------------------------------------------------------
double SubDomain::get_property(std::string name) const
{
  dolfin_error("SubDomain.cpp", "get parameter",
               "This method should be overloaded in the derived class");
  return 0.0;
}
//-----------------------------------------------------------------------------
