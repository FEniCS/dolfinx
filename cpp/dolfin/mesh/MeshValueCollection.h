// Copyright (C) 2011-2013 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
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
class MeshValueCollection
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

  /// Create a mesh value collection of entities of given dimension
  /// on a given mesh from topological data (cells) speifying entities
  /// to be marked and values_data specifying the marker.
  ///
  /// @param    mesh (_Mesh_)
  ///         The mesh associated with the collection.
  /// @param    dim (std::size_t)
  ///         The mesh entity dimension for the mesh value collection.
  /// @param    cells (std::vector<std::vector<T>>)
  ///         Array of cells to be marked (containing the global point
  ///         indices for each cell)
  /// @param    values_data (std::vector<T>)
  ///         Array of marker corresponding to each cell. This must be
  ///         same size as the number of rows in cells.
  MeshValueCollection(std::shared_ptr<const Mesh> mesh,
                    std::size_t dim,
                    std::vector<std::vector<T>>& cells,
                    std::vector<T>& values_data);

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

  /// Name
  std::string name = "f";

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
    : _mesh(mesh), _dim(dim)
{
  // Do nothing
}


template <typename T>
MeshValueCollection<T>::MeshValueCollection(
                      std::shared_ptr<const Mesh> mesh,
                      std::size_t dim,
                      std::vector<std::vector<T>>& cells,
                      std::vector<T>& values_data): _mesh(mesh), _dim(dim)
{ 
  // Ensure the mesh dimension is initialised. 
  // If the mesh is created from arrays, entities 
  // of all dimesions are not initialized.
  _mesh->create_entities(dim);

  const std::size_t D = _mesh->topology().dim();

  // Handle cells and vertex as a special case
  if (((int)D == _dim) || (_dim == 0))
  {
    for (std::size_t cell_index = 0; cell_index < values_data.size();
         ++cell_index)
    {
      const std::pair<std::size_t, std::size_t> key(cell_index, 0);
      _values.insert({key, values_data[cell_index]});
    }
  }
  else
  {
    // Number of vertices per cell is equal to the size of first
    // array of connectivity.
    const std::int32_t num_vertices_per_cell 
          = _mesh->topology().connectivity(_dim, 0)->size(0);
    std::vector<std::int32_t> v(num_vertices_per_cell);

    // Map from {entity vertex indices} to entity index
    std::map<std::vector<int>, size_t> entity_map;

    const std::vector<std::int64_t>& global_indices
        = _mesh->topology().global_indices(0);

    // Loop over all the entities of dimension _dim
    for (auto& m : mesh::MeshRange(*mesh, _dim))
    { 
      if (_dim == 0)
        v[0] = global_indices[m.index()];
      else
      {
        v.clear();
        for (auto& vtx : mesh::EntityRange(m, 0)){
          v.push_back(global_indices[vtx.index()]);
        }
        std::sort(v.begin(), v.end());
      }
      // The vector of vertex number is key and entity index is value
      entity_map[v] = m.index();
    }
    _mesh->create_connectivity(_dim, D);

    assert(_mesh->topology().connectivity(_dim, D));
    const Connectivity& connectivity = *_mesh->topology().connectivity(_dim, D);

    for (std::size_t j = 0; j < values_data.size();
         ++j)
    {
      // Find the cell
      v.clear();
      // cells[j] is a vector of length num_vertices_per_cell
      for (std::size_t i = 0; i < num_vertices_per_cell; ++i){
        v.push_back(cells[j][i]);
      }
      std::sort(v.begin(), v.end());

      auto map_it = entity_map.find(v);
      std::size_t entity_index = map_it->second;
      assert(connectivity.size(entity_index) > 0);

      const MeshEntity entity(*_mesh, _dim, entity_index);
      for (std::size_t i = 0; i < connectivity.size(entity_index); ++i)
      {
        // Create cell
        const mesh::MeshEntity cell(*_mesh, D,
                              connectivity.connections(entity_index)[i]);

        // Find the local entity index
        const std::size_t local_entity = cell.index(entity);

        // Insert into map
        const std::pair<std::size_t, std::size_t> key(cell.index(),
                                                      local_entity);
        _values.insert({key, values_data[j]});
      }
    }
  }
}
//---------------------------------------------------------------------------
template <typename T>
MeshValueCollection<T>::MeshValueCollection(
    const MeshFunction<T>& mesh_function)
    : _mesh(mesh_function.mesh()), _dim(mesh_function.dim())
{
  assert(_mesh);
  const int D = _mesh->topology().dim();

  // Prefetch values of mesh function
  Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, 1>> mf_values
      = mesh_function.values();

  // Handle cells as a special case
  if (D == _dim)
  {
    for (Eigen::Index cell_index = 0; cell_index < mf_values.size();
         ++cell_index)
    {
      const std::pair<std::size_t, std::size_t> key(cell_index, 0);
      _values.insert({key, mf_values[cell_index]});
    }
  }
  else
  {
    _mesh->create_connectivity(_dim, D);
    const Connectivity& connectivity = _mesh->topology().connectivity(_dim, D);
    for (Eigen::Index entity_index = 0; entity_index < mf_values.size();
         ++entity_index)
    {
      // Find the cell
      assert(connectivity.size(entity_index) > 0);
      const MeshEntity entity(*_mesh, _dim, entity_index);
      for (int i = 0; i < connectivity.size(entity_index); ++i)
      {
        // Create cell
        const mesh::MeshEntity cell(*_mesh, D,
                                    connectivity.connections(entity_index)[i]);

        // Find the local entity index
        const std::size_t local_entity = cell.index(entity);

        // Insert into map
        const std::pair<std::size_t, std::size_t> key(cell.index(),
                                                      local_entity);
        _values.insert({key, mf_values[entity_index]});
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
  const int D = _mesh->topology().dim();

  // FIXME: Use iterators

  // Prefetch values of mesh function
  Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, 1>> mf_values
      = mesh_function.values();

  // Handle cells as a special case
  if (D == _dim)
  {
    for (Eigen::Index cell_index = 0; cell_index < mf_values.size();
         ++cell_index)
    {
      const std::pair<std::size_t, std::size_t> key(cell_index, 0);
      _values.insert({key, mf_values[cell_index]});
    }
  }
  else
  {
    _mesh->create_connectivity(_dim, D);
    assert(_mesh->topology().connectivity(_dim, D));
    const Connectivity& connectivity = *_mesh->topology().connectivity(_dim, D);
    for (Eigen::Index entity_index = 0; entity_index < mf_values.size();
         ++entity_index)
    {
      // Find the cell
      assert(connectivity.size(entity_index) > 0);
      const MeshEntity entity(*_mesh, _dim, entity_index);
      for (std::size_t i = 0; i < connectivity.size(entity_index); ++i)
      {
        // Create cell
        const mesh::MeshEntity cell(*_mesh, D,
                                    connectivity.connections(entity_index)[i]);

        // Find the local entity index
        const std::size_t local_entity = cell.index(entity);

        // Insert into map
        const std::pair<std::size_t, std::size_t> key(cell.index(),
                                                      local_entity);
        _values.insert({key, mf_values[entity_index]});
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
    throw std::runtime_error(
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
    throw std::runtime_error(
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
  _mesh->create_connectivity(_dim, D);
  assert(_mesh->topology().connectivity(_dim, D));
  const Connectivity& connectivity = *_mesh->topology().connectivity(_dim, D);

  // Find the cell
  assert(connectivity.size(entity_index) > 0);
  const MeshEntity entity(*_mesh, _dim, entity_index);
  const mesh::MeshEntity cell(
      *_mesh, D, connectivity.connections(entity_index)[0]); // choose first

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
    throw std::runtime_error(
        "No value stored for cell index: " + std::to_string(cell_index)
        + " and local index: " + std::to_string(local_entity));
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

    // Verbose output of MeshValueCollection must be implemented manually.
  }
  else
  {
    s << "<MeshValueCollection of topological dimension " << dim()
      << " containing " << size() << " values>";
  }

  return s.str();
}
//---------------------------------------------------------------------------
} // namespace mesh
} // namespace dolfin
