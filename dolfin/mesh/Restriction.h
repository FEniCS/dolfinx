// Copyright (C) 2012 Anders Logg
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
// First added:  2012-11-14
// Last changed: 2012-11-14

#ifndef __RESTRICTION_H
#define __RESTRICTION_H

#include <boost/shared_ptr.hpp>

#include "MeshFunction.h"

namespace dolfin
{

  // Forward declarations
  class SubDomain;

  /// This class represents a restriction of a mesh to a subdomain,
  /// which can be defined as a subset of all the cells, the facets,
  /// or possibly lower dimensional entities of the mesh.

  class Restriction
  {
  public:

    /// Create cell-based restriction from subdomain
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh
    ///     sub_domain (_SubDomain_)
    ///         Sub domain defining the restriction
    Restriction(const Mesh& mesh, const SubDomain& sub_domain);

    /// Create restriction from subdomain to entities of arbitrary dimension
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh
    ///     sub_domain (_SubDomain_)
    ///         Sub domain defining the restriction
    ///     dim (uint)
    ///         Dimension of restriction
    Restriction(const Mesh& mesh, const SubDomain& sub_domain, uint dim);

    /// Create restriction from domain markers
    ///
    /// *Arguments*
    ///     domain_markers (_MeshFunction_ <uint>)
    ///         Domain markers for the cells of the mesh.
    ///     domain_number (uint)
    ///         Identifier for domain.
    Restriction(const MeshFunction<uint>& domain_markers,
                uint domain_number);

    /// Create restriction from domain markers (shared pointer version)
    ///
    /// *Arguments*
    ///     domain_markers (_MeshFunction_ <uint>)
    ///         Domain markers for the cells of the mesh.
    ///     domain_number (uint)
    ///         Identifier for domain.
    Restriction(boost::shared_ptr<const MeshFunction<uint> > domain_markers,
                uint domain_number);

    /// Return the full unrestricted mesh
    const Mesh& mesh() const;

    /// Return topological dimension of restriction
    uint dim() const;

    /// Check whether restriction contains entity
    bool contains(const MeshEntity& entity) const
    {
      dolfin_assert(_domain_markers);
      return (*_domain_markers)[entity] == _domain_number;
    }


  private:

    // Initialize domain markers from subdomain
    void init_from_subdomain(const Mesh& mesh,
                             const SubDomain& sub_domain, uint dim);

    // Domain markers
    boost::shared_ptr<const MeshFunction<uint> > _domain_markers;

    // Identifier for domain
    uint _domain_number;

  };

}

#endif
