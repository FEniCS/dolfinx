// Copyright (C) 2007-2013 Anders Logg
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
// First added:  2007-04-10
// Last changed: 2013-04-12

#ifndef __SUB_DOMAIN_H
#define __SUB_DOMAIN_H

#include <cstddef>
#include <map>
#include <dolfin/common/constants.h>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  template <typename T> class MeshFunction;
  template <typename T> class MeshValueCollection;
  template <typename T> class Array;

  /// This class defines the interface for definition of subdomains.
  /// Alternatively, subdomains may be defined by a _Mesh_ and a
  /// MeshFunction<std::size_t> over the mesh.

  class SubDomain
  {
  public:

    /// Constructor
    ///
    /// *Arguments*
    ///     map_tol (double)
    ///         The tolerance used when identifying mapped points using
    ///         the function SubDomain::map.
    SubDomain(const double map_tol=1.0e-10);

    /// Destructor
    virtual ~SubDomain();

    /// Return true for points inside the subdomain
    ///
    /// @param    x (Array<double>)
    ///         The coordinates of the point.
    /// @param   on_boundary (bool)
    ///         True for points on the boundary.
    ///
    /// @return    bool
    ///         True for points inside the subdomain.
    virtual bool inside(const Array<double>& x, bool on_boundary) const;

    /// Map coordinate x in domain H to coordinate y in domain G (used for
    /// periodic boundary conditions)
    ///
    /// @param   x (Array<double>)
    ///         The coordinates in domain H.
    /// @param    y (Array<double>)
    ///         The coordinates in domain G.
    virtual void map(const Array<double>& x, Array<double>& y) const;

    /// Snap coordinate to boundary of subdomain
    ///
    /// @param x (Array<double>)
    ///         The coordinates.
    virtual void snap(Array<double>& x) const {}

    //--- Marking of Mesh ---

    /// Set subdomain markers (std::size_t) on cells for given subdomain number
    ///
    /// @param    mesh (_Mesh_)
    ///         The mesh to be marked.
    /// @param    sub_domain (std::size_t)
    ///         The subdomain number.
    /// @param    check_midpoint (bool)
    ///         Flag for whether midpoint of cell should be checked (default).
    void mark_cells(Mesh& mesh,
                    std::size_t sub_domain,
                    bool check_midpoint=true) const;

    /// Set subdomain markers (std::size_t) on facets for given subdomain number
    ///
    /// @param    mesh (_Mesh_)
    ///         The mesh to be marked.
    /// @param    sub_domain (std::size_t)
    ///         The subdomain number.
    /// @param    check_midpoint (bool)
    ///         Flag for whether midpoint of cell should be checked (default).
    void mark_facets(Mesh& mesh,
                     std::size_t sub_domain,
                     bool check_midpoint=true) const;

    /// Set subdomain markers (std::size_t) for given topological dimension
    /// and subdomain number
    ///
    /// @param    mesh (_Mesh_)
    ///         The mesh to be marked.
    /// @param    dim (std::size_t)
    ///         The topological dimension of entities to be marked.
    /// @param    sub_domain (std::size_t)
    ///         The subdomain number.
    /// @param    check_midpoint (bool)
    ///         Flag for whether midpoint of cell should be checked (default).
    void mark(Mesh& mesh,
              std::size_t dim,
              std::size_t sub_domain,
              bool check_midpoint=true) const;

    //--- Marking of MeshFunction ---

    /// Set subdomain markers (std::size_t) for given subdomain number
    ///
    /// @param    sub_domains (MeshFunction<std::size_t>)
    ///         The subdomain markers.
    /// @param    sub_domain (std::size_t)
    ///         The subdomain number.
    /// @param    check_midpoint (bool)
    ///         Flag for whether midpoint of cell should be checked (default).
    void mark(MeshFunction<std::size_t>& sub_domains,
              std::size_t sub_domain,
              bool check_midpoint=true) const;

    /// Set subdomain markers (int) for given subdomain number
    ///
    /// @param    sub_domains (MeshFunction<int>)
    ///         The subdomain markers.
    /// @param    sub_domain (int)
    ///         The subdomain number.
    /// @param    check_midpoint (bool)
    ///         Flag for whether midpoint of cell should be checked (default).
    void mark(MeshFunction<int>& sub_domains,
              int sub_domain,
              bool check_midpoint=true) const;

    /// Set subdomain markers (double) for given subdomain number
    ///
    /// @param    sub_domains (MeshFunction<double>)
    ///         The subdomain markers.
    /// @param    sub_domain (double)
    ///         The subdomain number.
    /// @param    check_midpoint (bool)
    ///         Flag for whether midpoint of cell should be checked (default).
    void mark(MeshFunction<double>& sub_domains,
              double sub_domain,
              bool check_midpoint=true) const;

    /// Set subdomain markers (bool) for given subdomain
    ///
    /// @param    sub_domains (MeshFunction<bool>)
    ///         The subdomain markers.
    /// @param    sub_domain (bool)
    ///         The subdomain number.
    /// @param   check_midpoint (bool)
    ///         Flag for whether midpoint of cell should be checked (default).
    void mark(MeshFunction<bool>& sub_domains,
              bool sub_domain,
              bool check_midpoint=true) const;

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
              std::size_t sub_domain,
              const Mesh& mesh,
              bool check_midpoint=true) const;

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
    void mark(MeshValueCollection<int>& sub_domains,
              int sub_domain,
              const Mesh& mesh,
              bool check_midpoint=true) const;

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
    void mark(MeshValueCollection<double>& sub_domains,
              double sub_domain,
              const Mesh& mesh,
              bool check_midpoint=true) const;

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
    void mark(MeshValueCollection<bool>& sub_domains,
              bool sub_domain,
              const Mesh& mesh,
              bool check_midpoint=true) const;

    /// Return geometric dimension
    ///
    /// @return    std::size_t
    ///         The geometric dimension.
    std::size_t geometric_dimension() const;

    /// Return tolerance uses to find matching point via map function
    ///
    /// @return    double
    ///         The tolerance.
    const double map_tolerance;

  private:

    /// Apply marker of type T (most likely an std::size_t) to object of class
    /// S (most likely MeshFunction or MeshValueCollection)
    template<typename S, typename T>
    void apply_markers(S& sub_domains,
                       T sub_domain,
                       const Mesh& mesh,
                       bool check_midpoint) const;

    template<typename T>
      void apply_markers(std::map<std::size_t, std::size_t>& sub_domains,
                         std::size_t dim,
                         T sub_domain,
                         const Mesh& mesh,
                         bool check_midpoint) const;

    // Friends
    friend class DirichletBC;
    friend class PeriodicBC;

    // Geometric dimension, needed for SWIG interface, will be set before
    // calls to inside() and map()
    mutable std::size_t _geometric_dimension;

  };

}

#endif
