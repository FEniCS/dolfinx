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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2011-01-17
// Last changed: 2011-01-17

#ifndef __PARALLEL_DATA_H
#define __PARALLEL_DATA_H

#include <map>
#include <vector>
#include "MeshFunction.h"

namespace dolfin
{

  class Mesh;

  /// This class stores auxiliary mesh data for parallel computing.

  class ParallelData
  {
  public:

    /// Constructor
    ParallelData(const Mesh& mesh);

    /// Destructor
    //~ParallelData();

    //--- Data for distributed memory parallelism ---

    /// Return true if global indices have been computed for entity of
    /// dimension d
    bool have_global_entity_indices(uint d) const
    {
      if (_global_entity_indices.find(d) != _global_entity_indices.end())
        return true;
      else
        return false;
    }

    /// Return global indices for entity of dimension d
    MeshFunction<uint>& global_entity_indices(uint d)
    {
      if (!have_global_entity_indices(d))
        _global_entity_indices[d] = MeshFunction<uint>(mesh, d);
      return _global_entity_indices.find(d)->second;
    }

    /// Return global indices for entity of dimension d (const version)
    const MeshFunction<uint>& global_entity_indices(uint d) const
    {
      assert(have_global_entity_indices(d));
      return _global_entity_indices.find(d)->second;
    }

    /// FIXME: Add description and use better name
    std::map<uint, std::vector<uint> >& shared_vertices()
    { return _shared_vertices; }

    /// FIXME: Add description and use better name
    const std::map<uint, std::vector<uint> >& shared_vertices() const
    { return _shared_vertices; }

    MeshFunction<bool>& exterior_facet()
    { return _exterior_facet; }

    const MeshFunction<bool>& exterior_facet() const
    { return _exterior_facet; }

    std::vector<uint>& num_global_entities()
    { return _num_global_entities; }

    const std::vector<uint>& num_global_entities() const
    { return _num_global_entities; }

    //--- Data for shared memory parallelism (multicore) ---

    /*
    /// Return the number of colors for entities of dimension D
    /// colored by entities of dimension d and coloring distance rho.
    uint num_colors(uint D, uint d, uint rho) const;

    /// Return colors for entities of dimension D colored by entities
    /// of dimension d and coloring distance rho (const version).
    MeshFunction<uint>& entity_colors(uint D, uint d, uint rho);

    /// Return colors for entities of dimension D colored by entities
    /// of dimension d and coloring distance rho (const version).
    const MeshFunction<uint>& entity_colors(uint D, uint d, uint rho) const;

    /// Return an array of colored entities for each color in the
    /// range 0, 1, ..., num_colors -1 for entities of dimension D
    /// colored by entities of dimension d and coloring distance rho.
    std::vector<std::vector<uint > >& colored_entities(uint D, uint d, uint rho);

    /// Return an array of colored entities for each color in the
    /// range 0, 1, ..., num_colors for entities of dimension D
    /// colored by entities of dimension d and coloring distance rho
    /// (const version).
    const std::vector<std::vector<uint > >& colored_entities(uint D, uint d, uint rho) const;
    */

  private:

    // Mesh
    const Mesh& mesh;

    // Global indices for entity of dimension d
    std::map<uint, MeshFunction<unsigned int> > _global_entity_indices;

    // FIXME: Use better name
    // FIXME: Use unordered map?
    // FIXME: Use std::set instead of std::vector (the vector is sorted at some point in the code)

    // Maps each shared vertex to a list of the processes sharing
    // the vertex
    std::map<uint, std::vector<uint> > _shared_vertices;

    std::vector<uint> _num_global_entities;

    // True if a facet is an exterior facet, false otherwise
    MeshFunction<bool> _exterior_facet;

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
