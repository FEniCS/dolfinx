// Copyright (C) 2011 Anders Logg and Garth N. Wells
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
// First added:  2011-01-17
// Last changed: 2011-01-17

#ifndef __PARALLEL_DATA_H
#define __PARALLEL_DATA_H

#include <map>
#include <utility>
#include <vector>
#include <boost/scoped_ptr.hpp>
#include "dolfin/common/types.h"

namespace dolfin
{

  class Mesh;
  template<typename T> class MeshFunction;

  /// This class stores auxiliary mesh data for parallel computing.

  class ParallelData
  {
  public:

    /// Constructor
    ParallelData(const Mesh& mesh);

    /// Copy constructor
    ParallelData(const ParallelData& data);

    /// Destructor
    ~ParallelData();

    //--- Data for distributed memory parallelism ---

    /// Return true if global indices have been computed for entity of
    /// dimension d
    bool have_global_entity_indices(uint d) const;

    /// Return global indices (local-to-global) for entity of dimension d
    MeshFunction<uint>& global_entity_indices(uint d);

    /// Return global indices (local-to-global) for entity of dimension d (const version)
    const MeshFunction<uint>& global_entity_indices(uint d) const;

    /// Return global indices (local-to-global) for entity of dimension d in a vector
    std::vector<uint> global_entity_indices_as_vector(uint d) const;

    /// Return global-to-local indices for entity of dimension d
    const std::map<uint, uint>& global_to_local_entity_indices(uint d);

    /// Return global-to-local indices for entity of dimension d (const version)
    const std::map<uint, uint>& global_to_local_entity_indices(uint d) const;

    /// FIXME: Add description and use better name
    std::map<uint, std::vector<uint> >& shared_vertices();

    /// FIXME: Add description and use better name
    const std::map<uint, std::vector<uint> >& shared_vertices() const;

    /// Return MeshFunction that is true for globally exterior facets,
    /// false otherwise
    MeshFunction<bool>& exterior_facet();

    /// Return MeshFunction that is true for globally exterior facets,
    /// false otherwise (const version)
    const MeshFunction<bool>& exterior_facet() const;

    // Return the number of global entities of each dimension
    std::vector<uint>& num_global_entities();

    // Return the number of global entities of each dimension (const version)
    const std::vector<uint>& num_global_entities() const;


    //--- Data for shared memory parallelism (multicore) ---

    /// First vector is (colored entity dim - dim0 - .. -  colored entity dim).
    /// MeshFunction stores mesh entity colors and the vector<vector> is a list
    /// of all mesh entity indices of the same color,
    /// e.g. vector<vector>[col][i] is the index of the ith entity of
    /// color 'col'.
    std::map<const std::vector<uint>,
             std::pair<MeshFunction<uint>, std::vector<std::vector<uint> > > > coloring;

  private:

    // Mesh
    const Mesh& mesh;

    // For entity of dimension d, MeshFunction holding global indices
    std::map<uint, MeshFunction<unsigned int> > _global_entity_indices;

    // Global-to-local maps For entity of dimension d, MeshFunction holding global indices
    std::map<uint, std::map<uint, uint> > _global_to_local_entity_indices;

    // FIXME: Use better name
    // FIXME: Use unordered map?
    // FIXME: Use std::set instead of std::vector (the vector is sorted at some point in the code)

    // Maps each shared vertex to a list of the processes sharing
    // the vertex
    std::map<uint, std::vector<uint> > _shared_vertices;

    // Global number of entities of dimension d
    std::vector<uint> _num_global_entities;

    // True if a facet is an exterior facet, false otherwise
    boost::scoped_ptr<MeshFunction<bool> >_exterior_facet;

    /*
    // Some typedefs for complex types
    typedef boost::tuple<uint, uint, uint> tuple_type;
    typedef std::map<tuple_type, MeshFunction<uint> > entity_colors_map_type;
    typedef std::map<tuple_type, std::vector<std::vector<uint> > > colored_entities_map_type;

    // The mesh
    const Mesh& _mesh;

    // Map to entity colors
    entity_colors_map_type _entity_colors;

    // Map to colored entities
    colored_entities_map_type _colored_entities;

    */

  };

}

#endif
