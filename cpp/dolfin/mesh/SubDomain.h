// Copyright (C) 2007-2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Facet.h"
#include "MeshIterator.h"
#include "Vertex.h"
#include <Eigen/Dense>
#include <cstddef>
#include <dolfin/common/constants.h>
#include <dolfin/common/types.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshGeometry.h>
#include <map>

namespace dolfin
{
namespace mesh
{
class Mesh;
template <typename T>
class MeshFunction;
template <typename T>
class MeshValueCollection;

/// This class defines the interface for definition of subdomains.
/// Alternatively, subdomains may be defined by a _Mesh_ and a
/// MeshFunction<std::size_t> over the mesh.

class SubDomain
{
public:
  /// Constructor
  ///
  /// @param map_tol (double)
  ///         The tolerance used when identifying mapped points using
  ///         the function SubDomain::map.
  SubDomain(const double map_tol = 1.0e-10);

  /// Destructor
  virtual ~SubDomain();

  /// Return true for points inside the subdomain
  ///
  /// @param    x (Eigen::Ref<const Eigen::VectorXd>)
  ///         The coordinates of the point.
  /// @param   on_boundary (bool)
  ///         True for points on the boundary.
  ///
  /// @return    bool
  ///         True for points inside the subdomain.
  virtual EigenArrayXb inside(Eigen::Ref<const EigenRowArrayXXd> x,
                              bool on_boundary) const;

  /// Map coordinate x in domain H to coordinate y in domain G (used for
  /// periodic boundary conditions)
  ///
  /// @param   x (Eigen::Ref<const EigenArrayXd>)
  ///         The coordinates in domain H.
  /// @param    y (Eigen::Ref<EigenArrayXd>)
  ///         The coordinates in domain G.
  virtual void map(Eigen::Ref<const EigenArrayXd> x,
                   Eigen::Ref<EigenArrayXd> y) const;

  //--- Marking of MeshFunction ---

  /// Set subdomain markers (std::size_t) for given subdomain number
  ///
  /// @param    sub_domains (MeshFunction<std::size_t>)
  ///         The subdomain markers.
  /// @param    value (T)
  ///         The subdomain value.
  /// @param    check_midpoint (bool)
  ///         Flag for whether midpoint of cell should be checked (default).
  template <typename T>
  void mark(MeshFunction<T>& sub_domains, T value,
            bool check_midpoint = true) const
  {
    assert(sub_domains.mesh());
    mark(sub_domains, value, *sub_domains.mesh(), check_midpoint);
  }

  //--- Marking of MeshValueCollection ---

  /// Set subdomain markers (std::size_t) for given subdomain number
  ///
  /// @param    sub_domains (MeshValueCollection<std::size_t>)
  ///         The subdomain markers.
  /// @param    sub_domain (std::size_t)
  ///         The subdomain number.
  /// @param    mesh (_Mesh_)
  ///         The mesh.
  /// @param    check_midpoint (bool)
  ///         Flag for whether midpoint of cell should be checked (default).
  template <typename S, typename T>
  void mark(S& sub_domains, T sub_domain, const Mesh& mesh,
            bool check_midpoint) const;

  /// Return tolerance uses to find matching point via map function
  ///
  /// @return    double
  ///         The tolerance.
  const double map_tolerance;

private:
  template <typename T>
  void apply_markers(std::map<std::size_t, std::size_t>& sub_domains,
                     std::size_t dim, T sub_domain, const Mesh& mesh,
                     bool check_midpoint) const;
};

template <typename S, typename T>
void SubDomain::mark(S& sub_domains, T sub_domain, const Mesh& mesh,
                     bool check_midpoint) const
{
  log::log(TRACE, "Computing sub domain markers for sub domain %d.",
           sub_domain);

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
      {
        if (boundary_vertex[v[i]] == -1)
        {
          boundary_vertex[v[i]] = count;
          ++count;
        }
      }
    }
  }

  auto gdim = mesh.geometry().dim();

  // Check all vertices for "inside" (on_boundary==false)
  Eigen::Map<const EigenRowArrayXXd> x(mesh.geometry().x().data(),
                                       mesh.num_entities(0), gdim);
  EigenArrayXb all_inside = inside(x, false);
  assert(all_inside.rows() == x.rows());

  // Check all boundary vertices for "inside" (on_boundary==true)
  EigenRowArrayXXd x_bound(count, gdim);
  for (std::int32_t i = 0; i != mesh.num_entities(0); ++i)
  {
    if (boundary_vertex[i] != -1)
      x_bound.row(boundary_vertex[i]) = x.row(i);
  }
  EigenArrayXb bound_inside = inside(x_bound, true);
  assert(bound_inside.rows() == x_bound.rows());

  // Copy values back to vector, now -1="not on boundary anyway",
  // 1="inside", 0="not inside"
  for (std::int32_t i = 0; i != mesh.num_entities(0); ++i)
  {
    if (boundary_vertex[i] != -1)
      boundary_vertex[i] = bound_inside(boundary_vertex[i]) ? 1 : 0;
  }

  // Compute sub domain markers
  for (auto& entity : MeshRange<MeshEntity>(mesh, dim))
  {
    // An Entity is on_boundary if all its vertices are on the boundary
    bool on_boundary = true;

    // Assuming it is on boundary, also check if all points are "inside"
    bool all_points_inside = true;

    // Assuming it is not on boundary, check points in "all_inside"
    // array
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
      Eigen::Map<EigenRowArrayXd> x(entity.midpoint().coordinates(), gdim);
      if (!inside(x, on_boundary)[0])
        all_points_inside = false;
    }

    // Mark entity with all vertices inside
    if (all_points_inside)
      sub_domains[entity.index()] = sub_domain;
  }
}
//-----------------------------------------------------------------------------
}
}
