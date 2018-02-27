// Copyright (C) 2007-2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <dolfin/common/constants.h>
#include <dolfin/fem/DirichletBC.h>
#include <map>

using EigenRowMatrixXd
    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace dolfin
{

// Forward declarations
class Mesh;
template <typename T>
class MeshFunction;
template <typename T>
class MeshValueCollection;

namespace mesh
{

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
  virtual Eigen::Matrix<bool, Eigen::Dynamic, 1>
  inside(Eigen::Ref<const EigenRowMatrixXd> x, bool on_boundary) const;

  /// Map coordinate x in domain H to coordinate y in domain G (used for
  /// periodic boundary conditions)
  ///
  /// @param   x (Eigen::Ref<const Eigen::VectorXd>)
  ///         The coordinates in domain H.
  /// @param    y (Eigen::Ref<Eigen::VectorXd>)
  ///         The coordinates in domain G.
  virtual void map(Eigen::Ref<const Eigen::VectorXd> x,
                   Eigen::Ref<Eigen::VectorXd> y) const;

  //--- Marking of MeshFunction ---

  /// Set subdomain markers (std::size_t) for given subdomain number
  ///
  /// @param    sub_domains (MeshFunction<std::size_t>)
  ///         The subdomain markers.
  /// @param    sub_domain (std::size_t)
  ///         The subdomain number.
  /// @param    check_midpoint (bool)
  ///         Flag for whether midpoint of cell should be checked (default).
  void mark(MeshFunction<std::size_t>& sub_domains, std::size_t sub_domain,
            bool check_midpoint = true) const;

  /// Set subdomain markers (int) for given subdomain number
  ///
  /// @param    sub_domains (MeshFunction<int>)
  ///         The subdomain markers.
  /// @param    sub_domain (int)
  ///         The subdomain number.
  /// @param    check_midpoint (bool)
  ///         Flag for whether midpoint of cell should be checked (default).
  void mark(MeshFunction<int>& sub_domains, int sub_domain,
            bool check_midpoint = true) const;

  /// Set subdomain markers (double) for given subdomain number
  ///
  /// @param    sub_domains (MeshFunction<double>)
  ///         The subdomain markers.
  /// @param    sub_domain (double)
  ///         The subdomain number.
  /// @param    check_midpoint (bool)
  ///         Flag for whether midpoint of cell should be checked (default).
  void mark(MeshFunction<double>& sub_domains, double sub_domain,
            bool check_midpoint = true) const;

  /// Set subdomain markers (bool) for given subdomain
  ///
  /// @param    sub_domains (MeshFunction<bool>)
  ///         The subdomain markers.
  /// @param    sub_domain (bool)
  ///         The subdomain number.
  /// @param   check_midpoint (bool)
  ///         Flag for whether midpoint of cell should be checked (default).
  void mark(MeshFunction<bool>& sub_domains, bool sub_domain,
            bool check_midpoint = true) const;

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
  void mark(MeshValueCollection<std::size_t>& sub_domains,
            std::size_t sub_domain, const Mesh& mesh,
            bool check_midpoint = true) const;

  /// Set subdomain markers (int) for given subdomain number
  ///
  /// @param    sub_domains (MeshValueCollection<int>)
  ///         The subdomain markers
  /// @param    sub_domain (int)
  ///         The subdomain number
  /// @param  mesh (Mesh)
  ///         The mesh.
  /// @param    check_midpoint (bool)
  ///         Flag for whether midpoint of cell should be checked (default).
  void mark(MeshValueCollection<int>& sub_domains, int sub_domain,
            const Mesh& mesh, bool check_midpoint = true) const;

  /// Set subdomain markers (double) for given subdomain number
  ///
  /// @param    sub_domains (MeshValueCollection<double>)
  ///         The subdomain markers.
  /// @param    sub_domain (double)
  ///         The subdomain number
  /// @param  mesh (Mesh)
  ///         The mesh.
  /// @param    check_midpoint (bool)
  ///         Flag for whether midpoint of cell should be checked (default).
  void mark(MeshValueCollection<double>& sub_domains, double sub_domain,
            const Mesh& mesh, bool check_midpoint = true) const;

  /// Set subdomain markers (bool) for given subdomain
  ///
  /// @param     sub_domains (MeshValueCollection<bool>)
  ///         The subdomain markers
  /// @param    sub_domain (bool)
  ///         The subdomain number
  /// @param  mesh (Mesh)
  ///         The mesh.
  /// @param    check_midpoint (bool)
  ///         Flag for whether midpoint of cell should be checked (default).
  void mark(MeshValueCollection<bool>& sub_domains, bool sub_domain,
            const Mesh& mesh, bool check_midpoint = true) const;

  /// Return geometric dimension
  ///
  /// @return    std::size_t
  ///         The geometric dimension.
  std::size_t geometric_dimension() const;

  /// Property setter
  ///
  /// @param name
  /// @param value
  virtual void set_property(std::string name, double value);

  /// Property getter
  ///
  /// @param name
  /// @return double
  virtual double get_property(std::string name) const;

  /// Return tolerance uses to find matching point via map function
  ///
  /// @return    double
  ///         The tolerance.
  const double map_tolerance;

private:
  /// Apply marker of type T (most likely an std::size_t) to object of class
  /// S (most likely MeshFunction or MeshValueCollection)
  template <typename S, typename T>
  void apply_markers(S& sub_domains, T sub_domain, const Mesh& mesh,
                     bool check_midpoint) const;

  template <typename T>
  void apply_markers(std::map<std::size_t, std::size_t>& sub_domains,
                     std::size_t dim, T sub_domain, const Mesh& mesh,
                     bool check_midpoint) const;

  // Friends
  friend class dolfin::fem::DirichletBC;

  // Geometric dimension, needed for SWIG interface, will be set before
  // calls to inside() and map()
  mutable std::size_t _geometric_dimension;
};
}
}