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
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  template <typename T> class MeshFunction;
  template <typename T> class MeshValueCollection;

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
    MeshDomains(Mesh& mesh);

    /// Destructor
    ~MeshDomains();

    /// Value used for unset entities by default when converting to MeshFunctions
    static const std::size_t default_unset_value;

    /// Return maximum topological dimension of stored markers
    std::size_t max_dim() const;

    /// Return number of marked entities of given dimension
    std::size_t num_marked(std::size_t dim) const;

    /// Check whether domain data is empty
    bool is_empty() const;

    /// Get subdomain markers for given dimension (shared pointer version)
    std::map<std::size_t, std::size_t>& markers(std::size_t dim);

    /// Get subdomain markers for given dimension (const shared
    /// pointer version)
    const std::map<std::size_t, std::size_t>& markers(std::size_t dim) const;

    /// Return names of markers of a given dimension
    std::vector<std::string> marker_names(std::size_t dim) const;

    /// Get cell domains. This function computes the mesh function
    /// corresponding to markers of dimension D. The mesh function is
    /// cached for later access and will be computed on the first call
    /// to this function.
    boost::shared_ptr<const MeshFunction<std::size_t> >
      cell_domains(std::size_t unset_value=MeshDomains::default_unset_value) const;

    /// Get facet domains. This function computes the mesh function
    /// corresponding to markers of dimension D-1. The mesh function
    /// is cached for later access and will be computed on the first
    /// call to this function.
    boost::shared_ptr<const MeshFunction<std::size_t> >
      facet_domains(std::size_t unset_value=MeshDomains::default_unset_value) const;

    /// Create a mesh function corresponding to the MeshCollection 'collection'
    boost::shared_ptr<MeshFunction<std::size_t> >
      mesh_function(const MeshValueCollection<std::size_t>& collection,
		    std::size_t unset_value=MeshDomains::default_unset_value) const;

    /// Assignment operator
    const MeshDomains& operator= (const MeshDomains& domains);

    /// Initialize mesh domains for given topological dimension
    void init(std::size_t dim);

    /// Clear all data
    void clear();

  private:

    // The mesh
    Mesh& _mesh;

    // Subdomain markers for each geometric dim
    //std::vector<boost::shared_ptr<MeshValueCollection<std::size_t> > > _markers;
    //std::vector<std::vector<std::pair<std::size_t, std::size_t> > > _markers;
    std::vector<std::map<std::size_t, std::size_t> > _markers;

    // Named subdomain markers for each geometric dim
    std::vector<boost::unordered_map<std::string, boost::shared_ptr<MeshValueCollection<std::size_t> > > > _named_markers;

    // Mesh function for cell domains
    mutable boost::shared_ptr<MeshFunction<std::size_t> > _cell_domains;

    // Mesh function for facet domains (exterior or interior)
    mutable boost::shared_ptr<MeshFunction<std::size_t> > _facet_domains;

  };

}

#endif
