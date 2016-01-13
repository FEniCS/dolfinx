// Copyright (C) 2011-2013 Anders Logg and Garth N. Wells
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
// Modified by Chris Richardson, 2013.
//
// First added:  2006-08-30
// Last changed: 2013-05-22

#ifndef __MESH_VALUE_COLLECTION_H
#define __MESH_VALUE_COLLECTION_H

#include <map>
#include <utility>
#include <memory>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>
#include "Cell.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"

namespace dolfin
{

  /// The MeshValueCollection class can be used to store data
  /// associated with a subset of the entities of a mesh of a given
  /// topological dimension. It differs from the MeshFunction class in
  /// two ways. First, data does not need to be associated with all
  /// entities (only a subset). Second, data is associated with
  /// entities through the corresponding cell index and local entity
  /// number (relative to the cell), not by global entity index, which
  /// means that data may be stored robustly to file.

  template <typename T>
  class MeshValueCollection : public Variable
  {
  public:

    /// Create empty mesh value collection
    ///
    MeshValueCollection();

    /// Create an empty mesh value collection on a given mesh
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh.
    explicit MeshValueCollection(std::shared_ptr<const Mesh> mesh);

    /// Create a mesh value collection from a MeshFunction
    ///
    /// *Arguments*
    ///     mesh_function (_MeshFunction_ <T>)
    ///         The mesh function for creating a MeshValueCollection.
    explicit MeshValueCollection(const MeshFunction<T>& mesh_function);

    /// Create a mesh value collection of entities of given dimension
    /// on a given mesh
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh associated with the collection.
    ///     dim (std::size_t)
    ///         The mesh entity dimension for the mesh value collection.
    MeshValueCollection(const Mesh& mesh, std::size_t dim);

    /// Create a mesh value collection of entities of given dimension
    /// on a given mesh (shared_ptr version)
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh associated with the collection.
    ///     dim (std::size_t)
    ///         The mesh entity dimension for the mesh value collection.
    MeshValueCollection(std::shared_ptr<const Mesh> mesh, std::size_t dim);

    /// Create a mesh value collection from a file.
    ///
    /// *Arguments*
    ///     mesh (Mesh)
    ///         A mesh associated with the collection. The mesh is used to
    ///         map collection values to the appropriate process.
    ///     filename (std::string)
    ///         The XML file name.
    ///     dim (std::size_t)
    ///         The mesh entity dimension for the mesh value collection.
    MeshValueCollection(const Mesh& mesh, const std::string filename);

    /// Destructor
    ~MeshValueCollection() {}

    /// Assignment operator
    ///
    /// *Arguments*
    ///     mesh_function (_MeshFunction_)
    ///         A _MeshFunction_ object used to construct a
    ///         MeshValueCollection.
    MeshValueCollection<T>& operator=(const MeshFunction<T>& mesh_function);

    /// Assignment operator
    ///
    /// *Arguments*
    ///     mesh_value_collection (_MeshValueCollection_)
    ///         A _MeshValueCollection_ object used to construct a
    ///         MeshValueCollection.
    MeshValueCollection<T>&
      operator=(const MeshValueCollection<T>& mesh_value_collection);

    /// Initialise MeshValueCollection with mesh and dimension
    ///
    /// *Arguments*
    ///     mesh (_mesh))
    ///         The mesh on which the value collection is defined
    ///     dim (std::size_t)
    ///         The mesh entity dimension for the mesh value collection.
    void init(const Mesh& mesh, std::size_t dim);

    /// Initialise MeshValueCollection with mesh and dimension
    /// (shared_ptr version)
    ///
    /// *Arguments*
    ///     mesh (_mesh))
    ///         The mesh on which the value collection is defined
    ///     dim (std::size_t)
    ///         The mesh entity dimension for the mesh value collection.
    void init(std::shared_ptr<const Mesh> mesh, std::size_t dim);

    /// Set dimension. This function should not generally be used. It is
    /// for reading MeshValueCollections as the dimension is not
    /// generally known at construction.
    ///
    /// *Arguments*
    ///     dim (std::size_t)
    ///         The mesh entity dimension for the mesh value collection.
    void init(std::size_t dim);

    /// Return topological dimension
    ///
    /// *Returns*
    ///     std::size_t
    ///         The dimension.
    std::size_t dim() const;

    /// Return associated mesh
    ///
    /// *Returns*
    ///     _Mesh_
    ///         The mesh.
    std::shared_ptr<const Mesh> mesh() const;

    /// Return true if the subset is empty
    ///
    /// *Returns*
    ///     bool
    ///         True if the subset is empty.
    bool empty() const;

    /// Return size (number of entities in subset)
    ///
    /// *Returns*
    ///     std::size_t
    ///         The size.
    std::size_t size() const;

    /// Set marker value for given entity defined by a cell index and
    /// a local entity index
    ///
    /// *Arguments*
    ///     cell_index (std::size_t)
    ///         The index of the cell.
    ///     local_entity (std::size_t)
    ///         The local index of the entity relative to the cell.
    ///     marker_value (T)
    ///         The value of the marker.
    ///
    /// *Returns*
    ///     bool
    ///         True is a new value is inserted, false if overwriting
    ///         an existing value.
    bool set_value(std::size_t cell_index, std::size_t local_entity,
                   const T& value);

    /// Set value for given entity index
    ///
    /// *Arguments*
    ///     entity_index (std::size_t)
    ///         Index of the entity.
    ///     value (T).
    ///         The value of the marker.
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///
    /// *Returns*
    ///     bool
    ///         True is a new value is inserted, false if overwriting
    ///         an existing value.
    bool set_value(std::size_t entity_index, const T& value);

    /// Get marker value for given entity defined by a cell index and
    /// a local entity index
    ///
    /// *Arguments*
    ///     cell_index (std::size_t)
    ///         The index of the cell.
    ///     local_entity (std::size_t)
    ///         The local index of the entity relative to the cell.
    ///
    /// *Returns*
    ///     marker_value (T)
    ///         The value of the marker.
    T get_value(std::size_t cell_index, std::size_t local_entity);

    /// Get all values
    ///
    /// *Returns*
    ///     std::map<std::pair<std::size_t, std::size_t>, T>
    ///         A map from positions to values.
    std::map<std::pair<std::size_t, std::size_t>, T>& values();

    /// Get all values (const version)
    ///
    /// *Returns*
    ///     std::map<std::pair<std::size_t, std::size_t>, T>
    ///         A map from positions to values.
    const std::map<std::pair<std::size_t, std::size_t>, T>& values() const;

    /// Clear all values
    void clear();

    /// Return informal string representation (pretty-print)
    ///
    /// *Arguments*
    ///     verbose (bool)
    ///         Flag to turn on additional output.
    ///
    /// *Returns*
    ///     std::string
    ///         An informal representation.
    std::string str(bool verbose) const;

  private:

    // Associated mesh
    std::shared_ptr<const Mesh> _mesh;

    // Topological dimension
    int _dim;

    // The values
    std::map<std::pair<std::size_t, std::size_t>, T> _values;

  };

  //---------------------------------------------------------------------------
  // Implementation of MeshValueCollection
  //---------------------------------------------------------------------------
  template <typename T>
  MeshValueCollection<T>::MeshValueCollection()
    : Variable("m", "unnamed MeshValueCollection"), _dim(-1)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshValueCollection<T>::MeshValueCollection(std::shared_ptr<const Mesh>
                                              mesh) : _mesh(mesh), _dim(-1)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshValueCollection<T>::MeshValueCollection(const Mesh& mesh,
                                              std::size_t dim)
    : Variable("m", "unnamed MeshValueCollection"),
      _mesh(reference_to_no_delete_pointer(mesh)), _dim(dim)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshValueCollection<T>::MeshValueCollection(std::shared_ptr<const Mesh>
                                              mesh, std::size_t dim)
    : Variable("m", "unnamed MeshValueCollection"), _mesh(mesh), _dim(dim)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshValueCollection<T>::MeshValueCollection(const MeshFunction<T>&
                                              mesh_function)
    : Variable("m", "unnamed MeshValueCollection"), _mesh(mesh_function.mesh()),
      _dim(mesh_function.dim())
  {
    dolfin_assert(_mesh);
    const std::size_t D = _mesh->topology().dim();

    // Handle cells as a special case
    if ((int) D == _dim)
    {
      for (std::size_t cell_index = 0; cell_index < mesh_function.size();
           ++cell_index)
      {
        const std::pair<std::size_t, std::size_t> key(cell_index, 0);
        _values.insert({key, mesh_function[cell_index]});
      }
    }
    else
    {
      _mesh->init(_dim, D);
      const MeshConnectivity& connectivity = _mesh->topology()(_dim, D);
      dolfin_assert(!connectivity.empty());
      for (std::size_t entity_index = 0; entity_index < mesh_function.size();
           ++entity_index)
      {
        // Find the cell
        dolfin_assert(connectivity.size(entity_index) > 0);
        const MeshEntity entity(*_mesh, _dim, entity_index);
        for (std::size_t i = 0; i < entity.num_entities(D) ; ++i)
        {
          // Create cell
          const Cell cell(*_mesh, connectivity(entity_index)[i]);

          // Find the local entity index
          const std::size_t local_entity = cell.index(entity);

          // Insert into map
          const std::pair<std::size_t, std::size_t> key(cell.index(),
                                                        local_entity);
          _values.insert({key, mesh_function[entity_index]});
        }
      }
    }
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshValueCollection<T>::MeshValueCollection(const Mesh& mesh,
                                              const std::string filename)
    : Variable("m", "unnamed MeshValueCollection"),
      _mesh(reference_to_no_delete_pointer(mesh)), _dim(-1)
  {
    File file(filename);
    file >> *this;
    dolfin_assert(_dim > -1);
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshValueCollection<T>&
  MeshValueCollection<T>::operator=(const MeshFunction<T>& mesh_function)
  {
    _mesh = mesh_function.mesh();
    _dim = mesh_function.dim();

    dolfin_assert(_mesh);
    const std::size_t D = _mesh->topology().dim();

    // FIXME: Use iterators

    // Handle cells as a special case
    if ((int) D == _dim)
    {
      for (std::size_t cell_index = 0; cell_index < mesh_function.size();
           ++cell_index)
      {
        const std::pair<std::size_t, std::size_t> key(cell_index, 0);
        _values.insert({key, mesh_function[cell_index]});
      }
    }
    else
    {
      _mesh->init(_dim, D);
      const MeshConnectivity& connectivity = _mesh->topology()(_dim, D);
      dolfin_assert(!connectivity.empty());
      for (std::size_t entity_index = 0; entity_index < mesh_function.size();
           ++entity_index)
      {
        // Find the cell
        dolfin_assert(connectivity.size(entity_index) > 0);
        const MeshEntity entity(*_mesh, _dim, entity_index);
        for (std::size_t i = 0; i < entity.num_entities(D) ; ++i)
        {
          // Create cell
          const Cell cell(*_mesh, connectivity(entity_index)[i]);

          // Find the local entity index
          const std::size_t local_entity = cell.index(entity);

          // Insert into map
          const std::pair<std::size_t, std::size_t> key(cell.index(),
                                                        local_entity);
          _values.insert({key, mesh_function[entity_index]});
        }
      }
    }

    return *this;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshValueCollection<T>&
  MeshValueCollection<T>::operator=(const MeshValueCollection<T>&
                                    mesh_value_collection)
  {
    _mesh = mesh_value_collection._mesh;
    _dim = mesh_value_collection.dim();
    _values = mesh_value_collection.values();

    return *this;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshValueCollection<T>::init(const Mesh& mesh, std::size_t dim)
  {
    mesh.init(dim);
    _mesh = reference_to_no_delete_pointer(mesh);
    _dim = dim;
    _values.clear();
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshValueCollection<T>::init(std::shared_ptr<const Mesh> mesh,
                                    std::size_t dim)
  {
    mesh->init(dim);
    _mesh = mesh;
    _dim = dim;
    _values.clear();
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshValueCollection<T>::init(std::size_t dim)
  {
    dolfin_assert(_mesh);
    dolfin_assert(_dim < 0);
    _dim = dim;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  std::size_t MeshValueCollection<T>::dim() const
  {
    dolfin_assert(_dim >= 0);
    return _dim;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  bool MeshValueCollection<T>::empty() const
  {
    return _values.empty();
  }
  //---------------------------------------------------------------------------
  template <typename T>
  std::size_t MeshValueCollection<T>::size() const
  {
    return _values.size();
  }
  //---------------------------------------------------------------------------
  template <typename T>
  std::shared_ptr<const Mesh> MeshValueCollection<T>::mesh() const
  {
    dolfin_assert(_mesh);
    return _mesh;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  bool MeshValueCollection<T>::set_value(std::size_t cell_index,
                                         std::size_t local_entity,
                                         const T& value)
  {
    dolfin_assert(_dim >= 0);
    if (!_mesh)
    {
      dolfin_error("MeshValueCollection.h",
                   "set value",
                   "A mesh has not been associated with this MeshValueCollection");
    }

    const std::pair<std::size_t, std::size_t> pos(cell_index, local_entity);
    std::pair<typename std::map<std::pair<std::size_t, std::size_t>, T>::iterator, bool>
      it = _values.insert({pos, value});

    // If an item with same key already exists the value has not been
    // set and we need to update it
    if (!it.second)
      it.first->second = value;

    return it.second;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  bool MeshValueCollection<T>::set_value(std::size_t entity_index,
                                         const T& value)
  {
    if (!_mesh)
    {
      dolfin_error("MeshValueCollection.h",
                   "set value",
                   "A mesh has not been associated with this MeshValueCollection");
    }

    dolfin_assert(_dim >= 0);

    // Special case when d = D
    const std::size_t D = _mesh->topology().dim();
    if (_dim == (int) D)
    {
      // Set local entity index to zero when we mark a cell
      const std::pair<std::size_t, std::size_t> pos(entity_index, 0);
      std::pair<typename std::map<std::pair<std::size_t,
        std::size_t>, T>::iterator, bool> it;
      it = _values.insert({pos, value});

      // If an item with same key already exists the value has not been
      // set and we need to update it
      if (!it.second)
        it.first->second = value;

      return it.second;
    }

    // Get mesh connectivity d --> D
    _mesh->init(_dim, D);
    const MeshConnectivity& connectivity = _mesh->topology()(_dim, D);

    // Find the cell
    dolfin_assert(!connectivity.empty());
    dolfin_assert(connectivity.size(entity_index) > 0);
    const MeshEntity entity(*_mesh, _dim, entity_index);
    const Cell cell(*_mesh, connectivity(entity_index)[0]); // choose first

    // Find the local entity index
    const std::size_t local_entity = cell.index(entity);

    // Add value
    const std::pair<std::size_t, std::size_t> pos(cell.index(), local_entity);
    std::pair<typename std::map<std::pair<std::size_t,
      std::size_t>, T>::iterator, bool> it;
    it = _values.insert({pos, value});

    // If an item with same key already exists the value has not been
    // set and we need to update it
    if (!it.second)
      it.first->second = value;

    return it.second;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  T MeshValueCollection<T>::get_value(std::size_t cell_index,
				      std::size_t local_entity)
  {
    dolfin_assert(_dim >= 0);

    const std::pair<std::size_t, std::size_t> pos(cell_index, local_entity);
    const typename std::map<std::pair<std::size_t,
      std::size_t>, T>::const_iterator
      it = _values.find(pos);

    if (it == _values.end())
    {
      dolfin_error("MeshValueCollection.h",
                   "extract value",
                   "No value stored for cell index: %d and local index: %d",
                   cell_index, local_entity);
    }

    return it->second;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  std::map<std::pair<std::size_t, std::size_t>, T>&
    MeshValueCollection<T>::values()
  {
    return _values;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  const std::map<std::pair<std::size_t, std::size_t>, T>&
  MeshValueCollection<T>::values() const
  {
    return _values;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshValueCollection<T>::clear()
  {
    _values.clear();
  }
  //---------------------------------------------------------------------------
  template <typename T>
  std::string MeshValueCollection<T>::str(bool verbose) const
  {
    std::stringstream s;
    if (verbose)
    {
      s << str(false) << std::endl << std::endl;
      warning("Verbose output of MeshValueCollection must be implemented manually.");
    }
    else
    {
      s << "<MeshValueCollection of topological dimension " << dim()
        << " containing " << size() << " values>";
    }

    return s.str();
  }
  //---------------------------------------------------------------------------

}

#endif
