// Copyright (C) 2011 Anders Logg
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
// Modified by Garth N. Wells, 2012
//
// First added:  2011-08-29
// Last changed: 2012-04-03

#ifndef __MESH_DOMAINS_H
#define __MESH_DOMAINS_H

#include <map>
#include <vector>

namespace dolfin
{

  /// The class _MeshDomains_ stores the division of a _Mesh_ into
  /// subdomains. For each topological dimension 0 <= d <= D, where D
  /// is the topological dimension of the _Mesh_, a set of integer
  /// markers are stored for a subset of the entities of dimension d,
  /// indicating for each entity in the subset the number of the
  /// subdomain. It should be noted that the subset does not need to
  /// contain all entities of any given dimension; entities not
  /// contained in the subset are "unmarked".

  class MeshDomains
  {
  public:

    /// Create empty mesh domains
    MeshDomains();

    /// Destructor
    ~MeshDomains();

    /// Return maximum topological dimension of stored markers
    std::size_t max_dim() const;

    /// Return number of marked entities of given dimension
    std::size_t num_marked(std::size_t dim) const;

    /// Check whether domain data is empty
    bool is_empty() const;

    /// Get subdomain markers for given dimension (shared pointer
    /// version)
    std::map<std::size_t, std::size_t>& markers(std::size_t dim);

    /// Get subdomain markers for given dimension (const shared
    /// pointer version)
    const std::map<std::size_t, std::size_t>& markers(std::size_t dim) const;

    /// Set marker (entity index, marker value) of a given dimension
    /// d. Returns true if a new key is inserted, false otherwise.
    bool set_marker(std::pair<std::size_t, std::size_t> marker,
                    std::size_t dim);

    /// Get marker (entity index, marker value) of a given dimension
    /// d. Throws an error if marker does not exist.
    std::size_t get_marker(std::size_t entity_index, std::size_t dim) const;

    /// Assignment operator
    const MeshDomains& operator= (const MeshDomains& domains);

    /// Initialize mesh domains for given topological dimension
    void init(std::size_t dim);

    /// Clear all data
    void clear();

  private:

    // Subdomain markers for each geometric dimension
    std::vector<std::map<std::size_t, std::size_t> > _markers;

  };

}

#endif
