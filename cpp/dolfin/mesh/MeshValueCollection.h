// Copyright (C) 2011-2013 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Cell.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>
#include <map>
#include <memory>
#include <utility>

namespace dolfin
{
namespace mesh
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
class MeshValueCollection : public common::Variable
{
public:
  /// Copy constructor
  MeshValueCollection(const MeshValueCollection<T>& mvc) = default;

  /// Move constructor
  MeshValueCollection(MeshValueCollection<T>&& mvc) = default;

  /// Create a mesh value collection from a MeshFunction
  ///
  /// @param    mesh_function (MeshFunction<T>)
  ///         The mesh function for creating a MeshValueCollection.
  explicit MeshValueCollection(const MeshFunction<T>& mesh_function);

  /// Create a mesh value collection of entities of given dimension
  /// on a given mesh
  ///
  /// @param    mesh (_Mesh_)
  ///         The mesh associated with the collection.
  /// @param    dim (std::size_t)
  ///         The mesh entity dimension for the mesh value collection.
  MeshValueCollection(std::shared_ptr<const Mesh> mesh, std::size_t dim);

  /// Destructor
  ~MeshValueCollection() = default;

  /// Assignment operator
  ///
  /// @param    mesh_value_collection (_MeshValueCollection_)
  ///         A _MeshValueCollection_ object used to construct a
  ///         MeshValueCollection.
  MeshValueCollection<T>&
  operator=(const MeshValueCollection<T>& mesh_value_collection)
      = default;

  /// Assignment operator
  ///
  /// @param    mesh_function (_MeshFunction_)
  ///         A _MeshFunction_ object used to construct a
  ///         MeshValueCollection.
  MeshValueCollection<T>& operator=(const MeshFunction<T>& mesh_function);

  /// Return topological dimension
  ///
  /// @return   std::size_t
  ///         The dimension.
  std::size_t dim() const;

  /// Return associated mesh
  ///
  /// @return    _Mesh_
  ///         The mesh.
  std::shared_ptr<const Mesh> mesh() const;

  /// Return true if the subset is empty
  ///
  /// @return   bool
  ///         True if the subset is empty.
  bool empty() const;

  /// Return size (number of entities in subset)
  ///
  /// @return    std::size_t
  ///         The size.
  std::size_t size() const;

  // FIXME: remove
  /// Set marker value for given entity defined by a cell index and
  /// a local entity index
  ///
  /// @param    cell_index (std::size_t)
  ///         The index of the cell.
  /// @param    local_entity (std::size_t)
  ///         The local index of the entity relative to the cell.
  /// @param    value (T)
  ///         The value of the marker.
  ///
  /// @return    bool
  ///         True is a new value is inserted, false if overwriting
  ///         an existing value.
  bool set_value(std::size_t cell_index, std::size_t local_entity,
                 const T& value);

  /// Set value for given entity index
  ///
  /// @param    entity_index (std::size_t)
  ///         Index of the entity.
  /// @param    value (T).
  ///         The value of the marker.
  ///
  /// @return    bool
  ///         True is a new value is inserted, false if overwriting
  ///         an existing value.
  bool set_value(std::size_t entity_index, const T& value);

  /// Get marker value for given entity defined by a cell index and
  /// a local entity index
  ///
  /// @param    cell_index (std::size_t)
  ///         The index of the cell.
  /// @param    local_entity (std::size_t)
  ///         The local index of the entity relative to the cell.
  ///
  /// @return    marker_value (T)
  ///         The value of the marker.
  T get_value(std::size_t cell_index, std::size_t local_entity);

  /// Get all values
  ///
  /// @return    std::map<std::pair<std::size_t, std::size_t>, T>
  ///         A map from positions to values.
  std::map<std::pair<std::size_t, std::size_t>, T>& values();

  /// Get all values (const version)
  ///
  /// @return    std::map<std::pair<std::size_t, std::size_t>, T>
  ///         A map from positions to values.
  const std::map<std::pair<std::size_t, std::size_t>, T>& values() const;

  /// Clear all values
  void clear();

  /// Return informal string representation (pretty-print)
  ///
  /// @param   verbose (bool)
  ///         Flag to turn on additional output.
  ///
  /// @return    std::string
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
MeshValueCollection<T>::MeshValueCollection(std::shared_ptr<const Mesh> mesh,
                                            std::size_t dim)
    : common::Variable("m"), _mesh(mesh),
      _dim(dim)
{
  // Do nothing
}
//---------------------------------------------------------------------------
template <typename T>
MeshValueCollection<T>::MeshValueCollection(
    const MeshFunction<T>& mesh_function)
    : common::Variable("m"),
      _mesh(mesh_function.mesh()), _dim(mesh_function.dim())
{
  assert(_mesh);
  const std::size_t D = _mesh->topology().dim();

  // Handle cells as a special case
  if ((int)D == _dim)
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
    const MeshConnectivity& connectivity
        = _mesh->topology().connectivity(_dim, D);
    assert(!connectivity.empty());
    for (std::size_t entity_index = 0; entity_index < mesh_function.size();
         ++entity_index)
    {
      // Find the cell
      assert(connectivity.size(entity_index) > 0);
      const MeshEntity entity(*_mesh, _dim, entity_index);
      for (std::size_t i = 0; i < entity.num_entities(D); ++i)
      {
        // Create cell
        const mesh::Cell cell(*_mesh, connectivity(entity_index)[i]);

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
MeshValueCollection<T>& MeshValueCollection<T>::
operator=(const MeshFunction<T>& mesh_function)
{
  _mesh = mesh_function.mesh();
  _dim = mesh_function.dim();

  assert(_mesh);
  const std::size_t D = _mesh->topology().dim();

  // FIXME: Use iterators

  // Handle cells as a special case
  if ((int)D == _dim)
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
    const MeshConnectivity& connectivity
        = _mesh->topology().connectivity(_dim, D);
    assert(!connectivity.empty());
    for (std::size_t entity_index = 0; entity_index < mesh_function.size();
         ++entity_index)
    {
      // Find the cell
      assert(connectivity.size(entity_index) > 0);
      const MeshEntity entity(*_mesh, _dim, entity_index);
      for (std::size_t i = 0; i < entity.num_entities(D); ++i)
      {
        // Create cell
        const mesh::Cell cell(*_mesh, connectivity(entity_index)[i]);

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
std::size_t MeshValueCollection<T>::dim() const
{
  assert(_dim >= 0);
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
  assert(_mesh);
  return _mesh;
}
//---------------------------------------------------------------------------
template <typename T>
bool MeshValueCollection<T>::set_value(std::size_t cell_index,
                                       std::size_t local_entity, const T& value)
{
  assert(_dim >= 0);
  if (!_mesh)
  {
    log::dolfin_error(
        "MeshValueCollection.h", "set value",
        "A mesh has not been associated with this MeshValueCollection");
  }

  const std::pair<std::size_t, std::size_t> pos(cell_index, local_entity);
  std::pair<typename std::map<std::pair<std::size_t, std::size_t>, T>::iterator,
            bool>
      it = _values.insert({pos, value});

  // If an item with same key already exists the value has not been
  // set and we need to update it
  if (!it.second)
    it.first->second = value;

  return it.second;
}
//---------------------------------------------------------------------------
template <typename T>
bool MeshValueCollection<T>::set_value(std::size_t entity_index, const T& value)
{
  if (!_mesh)
  {
    log::dolfin_error(
        "MeshValueCollection.h", "set value",
        "A mesh has not been associated with this MeshValueCollection");
  }

  assert(_dim >= 0);

  // Special case when d = D
  const std::size_t D = _mesh->topology().dim();
  if (_dim == (int)D)
  {
    // Set local entity index to zero when we mark a cell
    const std::pair<std::size_t, std::size_t> pos(entity_index, 0);
    std::pair<
        typename std::map<std::pair<std::size_t, std::size_t>, T>::iterator,
        bool>
        it;
    it = _values.insert({pos, value});

    // If an item with same key already exists the value has not been
    // set and we need to update it
    if (!it.second)
      it.first->second = value;

    return it.second;
  }

  // Get mesh connectivity d --> D
  _mesh->init(_dim, D);
  const MeshConnectivity& connectivity
      = _mesh->topology().connectivity(_dim, D);

  // Find the cell
  assert(!connectivity.empty());
  assert(connectivity.size(entity_index) > 0);
  const MeshEntity entity(*_mesh, _dim, entity_index);
  const mesh::Cell cell(*_mesh, connectivity(entity_index)[0]); // choose first

  // Find the local entity index
  const std::size_t local_entity = cell.index(entity);

  // Add value
  const std::pair<std::size_t, std::size_t> pos(cell.index(), local_entity);
  std::pair<typename std::map<std::pair<std::size_t, std::size_t>, T>::iterator,
            bool>
      it;
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
  assert(_dim >= 0);

  const std::pair<std::size_t, std::size_t> pos(cell_index, local_entity);
  const typename std::map<std::pair<std::size_t, std::size_t>,
                          T>::const_iterator it
      = _values.find(pos);

  if (it == _values.end())
  {
    log::dolfin_error("MeshValueCollection.h", "extract value",
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
    log::warning(
        "Verbose output of MeshValueCollection must be implemented manually.");
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
}