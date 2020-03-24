// Copyright (C) 2011-2020 Anders Logg, Garth N. Wells, Michal Habera and
// Abhinav Gupta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "MeshFunction.h"
#include <map>
#include <memory>
#include <utility>

namespace dolfinx
{
namespace mesh
{

/// The MeshValueCollection class can be used to store data associated
/// with a subset of the entities of a mesh of a given topological
/// dimension. It differs from the MeshFunction class in two ways.
/// First, data does not need to be associated with all entities (only a
/// subset). Second, data is associated with entities through the
/// corresponding cell index and local entity number (relative to the
/// cell), not by global entity index, which means that data may be
/// stored robustly to file.

template <typename T>
class MeshValueCollection
{
public:
  /// Copy constructor
  MeshValueCollection(const MeshValueCollection<T>& mvc) = default;

  /// Move constructor
  MeshValueCollection(MeshValueCollection<T>&& mvc) = default;

  /// Create a mesh value collection of entities of given dimension on a
  /// given mesh
  /// @param[in] mesh The mesh associated with the collection
  /// @param[in] dim The mesh entity dimension for the mesh value
  ///                collection.
  MeshValueCollection(std::shared_ptr<const Mesh> mesh, std::size_t dim);

  /// Create a mesh value collection of entities of given dimension
  /// on a given mesh from topological data speifying entities
  /// to be marked and data specifying the value of the marker.
  ///
  /// @param[in] mesh The mesh associated with the collection.
  /// @param[in] dim The mesh entity dimension for the mesh value collection.
  /// @param[in] cells An array describing topology of marked mesh entities.
  ///                  Use parallel-global vertex indices.
  /// @param[in] values_data An array of values attached to marked mesh
  ///                        entities. Size of this array must agree with
  ///                        number of columns in `cells`.
  MeshValueCollection(
      std::shared_ptr<const Mesh> mesh, int dim,
      const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                          Eigen::RowMajor>>& cells,
      const Eigen::Ref<const Eigen::Array<T, 1, Eigen::Dynamic,
                                          Eigen::RowMajor>>& values_data);

  /// Destructor
  ~MeshValueCollection() = default;

  /// Assignment operator
  /// @param[in] mesh_value_collection A MeshValueCollection object used
  ///                                  to construct a
  ///                                  MeshValueCollection.
  MeshValueCollection<T>&
  operator=(const MeshValueCollection<T>& mesh_value_collection)
      = default;

  /// Assignment operator from a MeshFunction
  /// @param[in] mesh_function A MeshFunction used to construct a
  ///                          MeshValueCollection.
  MeshValueCollection<T>& operator=(const MeshFunction<T>& mesh_function);

  /// Return topological dimension
  /// @return  The dimension
  std::size_t dim() const;

  /// Return associated mesh
  /// @return The mesh
  std::shared_ptr<const Mesh> mesh() const;

  /// Return true if the subset is empty
  /// @return True if the subset is empty.
  bool empty() const;

  /// Return size (number of entities in subset)
  /// @return  Number of mesh entities
  std::size_t size() const;

  // FIXME: remove
  /// Set marker value for given entity defined by a cell index and a
  /// local entity index
  ///
  /// @param[in] cell_index The index of the cell
  /// @param[in] local_entity The local index of the entity relative to
  ///                         the cell
  /// @param[in] value The value of the marker.
  /// @return True is a new value is inserted, false if overwriting an
  /// existing value.
  bool set_value(std::size_t cell_index, std::size_t local_entity,
                 const T& value);

  /// Set value for given entity index
  /// @param[in] entity_index Index of the entity
  /// @param[in] value The value of the marker.
  /// @return True if a new value is inserted, false if overwriting an
  ///         existing value
  bool set_value(std::size_t entity_index, const T& value);

  /// Get marker value for given entity defined by a cell index and a
  /// local entity index
  /// @param[in] cell_index The index of the cell
  /// @param[in] local_entity The local index of the entity relative to
  ///                         the cell.
  /// @return marker_value The value of the marker
  T get_value(std::size_t cell_index, std::size_t local_entity);

  /// Get all values
  /// @return A map from positions to values.
  std::map<std::pair<std::size_t, std::size_t>, T>& values();

  /// Get all values (const version)
  /// @return Map from positions to values.
  const std::map<std::pair<std::size_t, std::size_t>, T>& values() const;

  /// Clear all values
  void clear();

  /// Return informal string representation (pretty-print)
  /// @param[in] verbose Flag to turn on additional output
  /// @return An informal representation
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
  assert(mesh);
  const int D = mesh->topology().dim();
  mesh->create_connectivity(dim, D);
}
//---------------------------------------------------------------------------
template <typename T>
MeshValueCollection<T>::MeshValueCollection(
    std::shared_ptr<const Mesh> mesh, int dim,
    const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& cells,
    const Eigen::Ref<const Eigen::Array<T, 1, Eigen::Dynamic, Eigen::RowMajor>>&
        values_data)
    : _mesh(mesh), _dim(dim)
{

  if (cells.rows() != values_data.cols())
  {
    throw std::runtime_error(
        "Not matching number of mesh entities and data attached to it.");
  }

  // Ensure the mesh dimension is initialised. If the mesh is created
  // from arrays, entities of all dimesions are not initialized.
  _mesh->create_entities(dim);

  const int tdim = _mesh->topology().dim();

  // Handle cells and vertices as a special case
  if (tdim == _dim or _dim == 0)
  {
    for (Eigen::Index cell_index = 0; cell_index < values_data.cols();
         ++cell_index)
    {
      const std::pair<std::size_t, std::size_t> key(cell_index, 0);
      _values.insert({key, values_data(cell_index)});
    }
  }
  else
  {
    // Number of vertices per cell is equal to the size of first array
    // of connectivity
    auto e_to_v = _mesh->topology().connectivity(dim, 0);
    assert(e_to_v);

    // FIXME: this is bad - get number of vertices from cell type
    const int num_vertices_per_entity = e_to_v->num_links(0);
    std::vector<std::int64_t> v(num_vertices_per_entity);

    // Map from {entity vertex indices} to entity index
    std::map<std::vector<std::int64_t>, std::size_t> entity_map;

    auto vertices_map = _mesh->topology().index_map(0);
    assert(vertices_map);

    const std::vector<std::int64_t> global_indices
        = vertices_map->global_indices(false);

    auto entities_map = _mesh->topology().index_map(dim);
    assert(entities_map);

    const std::int32_t num_entities
        = entities_map->size_local() + entities_map->num_ghosts();

    // Loop over all the entities of dimension _dim
    for (std::int32_t i = 0; i < num_entities; ++i)
    {
      if (_dim == 0)
        v[0] = global_indices[i];
      else
      {
        auto entity_vertices = e_to_v->links(i);
        for (int j = 0; j < num_vertices_per_entity; ++j)
          v[j] = global_indices[entity_vertices[j]];
        std::sort(v.begin(), v.end());
      }
      entity_map.emplace_hint(entity_map.end(), v, i);
    }
    _mesh->create_connectivity(dim, tdim);

    auto entity_cell_connectivity = _mesh->topology().connectivity(dim, tdim);
    assert(entity_cell_connectivity);

    auto c_to_e = _mesh->topology().connectivity(tdim, dim);
    assert(c_to_e);

    // Get cell type for entity on which the MVC lives
    const mesh::CellType entity_cell_type
        = mesh::cell_entity_type(_mesh->topology().cell_type(), dim);

    // Number of nodes per entity is deduced from the size
    // of provided entity topology
    const int num_nodes_per_entity = cells.cols();

    // Fetch nodes-to-vertices mapping for the case of higher order
    // meshes
    const std::vector<int> nodes_to_verts
        = mesh::cell_vertex_indices(entity_cell_type, num_nodes_per_entity);

    for (Eigen::Index j = 0; j < values_data.cols(); ++j)
    {
      // Apply node to vertices mapping, this throws away
      // nodes read from the file
      for (int i = 0; i < num_vertices_per_entity; ++i)
        v[i] = cells(j, nodes_to_verts[i]);

      std::sort(v.begin(), v.end());

      // Find mesh entity given its vertex indices
      auto map_it = entity_map.find(v);
      if (map_it == entity_map.end())
        throw std::runtime_error("Entity not found in the mesh.");

      const std::size_t entity_index = map_it->second;

      // For this entity need to find all linked cells
      // and local index wrt. these cells
      auto entity_cells = entity_cell_connectivity->links(entity_index);
      assert(entity_cells.size() > 0);

      for (int i = 0; i < entity_cells.size(); ++i)
      {
        const int cell_index = entity_cells[i];
        auto cell_entities = c_to_e->links(cell_index);
        const auto it = std::find(cell_entities.data(),
                                  cell_entities.data() + cell_entities.size(),
                                  entity_index);
        const int local_entity = it - cell_entities.data();

        // Insert into map
        _values.emplace_hint(
            _values.end(),
            std::pair<std::size_t, std::size_t>(cell_index, local_entity),
            values_data(j));
      }
    }
  }
}
//---------------------------------------------------------------------------
template <typename T>
MeshValueCollection<T>&
MeshValueCollection<T>::operator=(const MeshFunction<T>& mesh_function)
{
  _mesh = mesh_function.mesh();
  _dim = mesh_function.dim();
  assert(_mesh);
  const int tdim = _mesh->topology().dim();

  // Prefetch values of mesh function
  const Eigen::Array<T, Eigen::Dynamic, 1>& mf_values = mesh_function.values();

  // Handle cells as a special case
  if (tdim == _dim)
  {
    for (Eigen::Index cell = 0; cell < mf_values.size(); ++cell)
      _values.insert({{cell, 0}, mf_values[cell]});
  }
  else
  {
    _mesh->create_connectivity(_dim, tdim);
    assert(_mesh->topology().connectivity(_dim, tdim));

    // Get mesh connectivity d -> tdim and  tdim -> d
    auto e_to_c = _mesh->topology().connectivity(_dim, tdim);
    if (!e_to_c)
      throw std::runtime_error("Missing connectivity.");
    auto c_to_e = _mesh->topology().connectivity(tdim, _dim);
    if (!c_to_e)
      throw std::runtime_error("Missing connectivity.");

    for (Eigen::Index entity = 0; entity < mf_values.size(); ++entity)
    {
      auto cells = e_to_c->links(entity);
      for (int i = 0; i < cells.rows(); ++i)
      {
        // Find the local entity index
        auto entities = c_to_e->links(cells[i]);
        auto p = std::find(entities.data(), entities.data() + entities.rows(),
                           entity);
        assert(p != (entities.data() + entities.rows()));
        const int local_entity = std::distance(entities.data(), p);

        // Insert into map
        _values.insert({{cells[i], local_entity}, mf_values[entity]});
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
  auto it = _values.insert({pos, value});

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

  // Special case when d = tdim
  const int tdim = _mesh->topology().dim();
  if (_dim == tdim)
  {
    // Set local entity index to zero when we mark a cell
    const std::pair<std::size_t, std::size_t> pos(entity_index, 0);
    auto it = _values.insert({pos, value});

    // If an item with same key already exists the value has not been
    // set and we need to update it
    if (!it.second)
      it.first->second = value;

    return it.second;
  }

  // Get mesh connectivity d -> tdim and  tdim -> d
  auto e_to_c = _mesh->topology().connectivity(_dim, tdim);
  if (!e_to_c)
    throw std::runtime_error("Missing connectivity.");
  auto c_to_e = _mesh->topology().connectivity(tdim, _dim);
  if (!c_to_e)
    throw std::runtime_error("Missing connectivity.");

  // Find the cell (pick first)
  assert(e_to_c->num_links(entity_index) > 0);
  const int cell = e_to_c->links(entity_index)[0];

  // Find the local entity index
  auto entities = c_to_e->links(cell);
  auto p = std::find(entities.data(), entities.data() + entities.rows(),
                     entity_index);
  assert(p != (entities.data() + entities.rows()));
  const int local_entity = std::distance(entities.data(), p);

  // Add value
  const std::pair<std::size_t, std::size_t> pos(cell, local_entity);
  auto it = _values.insert({pos, value});

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
  auto it = _values.find(pos);
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
} // namespace dolfinx
