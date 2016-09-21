// Copyright (C) 2006-2011 Anders Logg
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
// Modified by Andre Massing, 2009.
// Modified by Garth N. Wells, 2012.
//
// First added:  2006-05-11
// Last changed: 2014-07-02

#ifndef __MESH_ENTITY_H
#define __MESH_ENTITY_H

#include <cmath>
#include <iostream>

#include <dolfin/geometry/Point.h>
#include "Mesh.h"

namespace dolfin
{

  //class Mesh;
  class Point;

  /// A MeshEntity represents a mesh entity associated with
  /// a specific topological dimension of some _Mesh_.

  class MeshEntity
  {
  public:

    /// Default Constructor
    MeshEntity() : _mesh(0), _dim(0), _local_index(0) {}

    /// Constructor
    ///
    /// @param   mesh (_Mesh_)
    ///         The mesh.
    /// @param     dim (std::size_t)
    ///         The topological dimension.
    /// @param     index (std::size_t)
    ///         The index.
    MeshEntity(const Mesh& mesh, std::size_t dim, std::size_t index);

    /// Destructor
    virtual ~MeshEntity();

    /// Initialize mesh entity with given data
    ///
    /// @param      mesh (_Mesh_)
    ///         The mesh.
    /// @param     dim (std::size_t)
    ///         The topological dimension.
    /// @param     index (std::size_t)
    ///         The index.
    void init(const Mesh& mesh, std::size_t dim, std::size_t index);

    /// Comparison Operator
    ///
    /// @param e (MeshEntity)
    ///         Another mesh entity
    ///
    ///  @return    bool
    ///         True if the two mesh entities are equal.
    bool operator==(const MeshEntity& e) const
    {
      return (_mesh == e._mesh && _dim == e._dim
              && _local_index == e._local_index);
    }

    /// Comparison Operator
    ///
    /// @param e (MeshEntity)
    ///         Another mesh entity.
    ///
    /// @return     bool
    ///         True if the two mesh entities are NOT equal.
    bool operator!=(const MeshEntity& e) const
    { return !operator==(e); }

    /// Return mesh associated with mesh entity
    ///
    /// @return Mesh
    ///         The mesh.
    const Mesh& mesh() const
    { return *_mesh; }

    /// Return topological dimension
    ///
    /// @return     std::size_t
    ///         The dimension.
    std::size_t dim() const
    { return _dim; }

    /// Return index of mesh entity
    ///
    /// @return     std::size_t
    ///         The index.
    std::size_t index() const
    { return _local_index; }

    /// Return global index of mesh entity
    ///
    /// @return     std::size_t
    ///         The global index. Set to
    ///         std::numerical_limits<std::size_t>::max() if global index
    ///         has not been computed
    std::size_t global_index() const
    { return _mesh->topology().global_indices(_dim)[_local_index]; }

    /// Return local number of incident mesh entities of given
    /// topological dimension
    ///
    /// @param     dim (std::size_t)
    ///         The topological dimension.
    ///
    /// @return     std::size_t
    /// The number of local incident MeshEntity objects of given
    /// dimension.
    std::size_t num_entities(std::size_t dim) const
    { return _mesh->topology()(_dim, dim).size(_local_index); }

    /// Return global number of incident mesh entities of given
    /// topological dimension
    ///
    /// @param     dim (std::size_t)
    ///         The topological dimension.
    ///
    /// @return     std::size_t
    ///         The number of global incident MeshEntity objects of given
    ///         dimension.
    std::size_t num_global_entities(std::size_t dim) const
    { return _mesh->topology()(_dim, dim).size_global(_local_index); }

    /// Return array of indices for incident mesh entities of given
    /// topological dimension
    ///
    /// @param     dim (std::size_t)
    ///         The topological dimension.
    ///
    /// @return     std::size_t
    ///         The index for incident mesh entities of given dimension.
    const unsigned int* entities(std::size_t dim) const
    {
      const unsigned int* initialized_mesh_entities
        = _mesh->topology()(_dim, dim)(_local_index);
      dolfin_assert(initialized_mesh_entities);
      return initialized_mesh_entities;
    }

    /// Return unique mesh ID
    ///
    /// @return     std::size_t
    ///         The unique mesh ID.
    std::size_t mesh_id() const
    { return _mesh->id(); }

    /// Check if given entity is incident
    ///
    /// @param     entity (_MeshEntity_)
    ///         The entity.
    ///
    ///  @return    bool
    ///         True if the given entity is incident
    bool incident(const MeshEntity& entity) const;

    /// Compute local index of given incident entity (error if not
    /// found)
    ///
    /// @param     entity (_MeshEntity_)
    ///         The mesh entity.
    ///
    /// @return     std::size_t
    ///         The local index of given entity.
    std::size_t index(const MeshEntity& entity) const;

    /// Compute midpoint of cell
    ///
    /// @return Point
    ///         The midpoint of the cell.
    Point midpoint() const;

    /// Determine whether an entity is a 'ghost' from another
    /// process
    /// @return bool
    ///    True if entity is a ghost entity
    bool is_ghost() const
    { return (_local_index >= _mesh->topology().ghost_offset(_dim)); }

    /// Return set of sharing processes
    /// @return std::set<unsigned int>
    ///   List of sharing processes
    std::set<unsigned int> sharing_processes() const
    {
      const std::map<std::int32_t, std::set<unsigned int>>& sharing_map
        = _mesh->topology().shared_entities(_dim);
      const auto map_it = sharing_map.find(_local_index);
      if (map_it == sharing_map.end())
        return std::set<unsigned int>();
      else
        return map_it->second;
    }

    /// Determine if an entity is shared or not
    /// @return bool
    ///    True if entity is shared
    bool is_shared() const
    {
      if (_mesh->topology().have_shared_entities(_dim))
      {
        const std::map<std::int32_t, std::set<unsigned int>>& sharing_map
          = _mesh->topology().shared_entities(_dim);
        return (sharing_map.find(_local_index) != sharing_map.end());
      }
      return false;
    }

    /// Get ownership of this entity - only really valid for cells
    /// @return unsigned int
    ///    Owning process
    unsigned int owner() const;

    // Note: Not a subclass of Variable for efficiency!
    /// Return informal string representation (pretty-print)
    ///
    /// @param      verbose (bool)
    ///         Flag to turn on additional output.
    ///
    /// @return      std::string
    ///         An informal representation of the function space.
    std::string str(bool verbose) const;

  protected:

    // Friends
    friend class MeshEntityIterator;
    template<typename T> friend class MeshEntityIteratorBase;
    friend class SubsetIterator;

    // The mesh
    Mesh const * _mesh;

    // Topological dimension
    std::size_t _dim;

    // Local index of entity within topological dimension
    std::size_t _local_index;

  };

}

#endif
