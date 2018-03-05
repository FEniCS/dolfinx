// Copyright (C) 2006-2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "LocalMeshValueCollection.h"
#include "Mesh.h"
#include "MeshConnectivity.h"
#include "MeshEntity.h"
#include <boost/container/vector.hpp>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>
#include <map>
#include <memory>
#include <unordered_set>

namespace dolfin
{
namespace mesh
{
class MeshEntity;

/// A MeshFunction is a function that can be evaluated at a set of
/// mesh entities. A MeshFunction is discrete and is only defined at
/// the set of mesh entities of a fixed topological dimension.  A
/// MeshFunction may for example be used to store a global numbering
/// scheme for the entities of a (parallel) mesh, marking sub
/// domains or boolean markers for mesh refinement.

template <typename T>
class MeshFunction : public common::Variable
{
public:
  /// Create mesh of given dimension on given mesh and initialize
  /// to a value
  ///
  /// @param     mesh (_Mesh_)
  ///         The mesh to create mesh function on.
  /// @param     dim (std::size_t)
  ///         The mesh entity dimension.
  /// @param     value (T)
  ///         The value.
  MeshFunction(std::shared_ptr<const Mesh> mesh, std::size_t dim,
               const T& value);

  /// Create function from a MeshValueCollecion (shared_ptr version)
  ///
  /// @param mesh (_Mesh_)
  ///         The mesh to create mesh function on.
  /// @param value_collection (_MeshValueCollection_)
  ///         The mesh value collection for the mesh function data.
  MeshFunction(std::shared_ptr<const Mesh> mesh,
               const MeshValueCollection<T>& value_collection);

  /// Copy constructor
  ///
  /// @param f (_MeshFunction_)
  ///         The object to be copied.
  MeshFunction(const MeshFunction<T>& f) = default;

  /// Move constructor
  ///
  /// @param f (_MeshFunction_)
  ///         The object to be moved.
  MeshFunction(MeshFunction<T>&& f) = default;

  /// Destructor
  ~MeshFunction() = default;

  /// Assign mesh function to other mesh function
  /// Assignment operator
  ///
  /// @param f (_MeshFunction_)
  ///         A _MeshFunction_ object to assign to another MeshFunction.
  MeshFunction<T>& operator=(const MeshFunction<T>& f) = default;

  /// Assignment operator
  ///
  /// @param mesh (_MeshValueCollection_)
  ///         A _MeshValueCollection_ object used to construct a MeshFunction.
  MeshFunction<T>& operator=(const MeshValueCollection<T>& mesh);

  /// Return mesh associated with mesh function
  ///
  /// @return _Mesh_
  ///         The mesh.
  std::shared_ptr<const Mesh> mesh() const;

  /// Return topological dimension
  ///
  /// @return std::size_t
  ///         The dimension.
  std::size_t dim() const;

  /// Return size (number of entities)
  ///
  /// @return std::size_t
  ///         The size.
  std::size_t size() const;

  /// Return array of values (const. version)
  ///
  /// return T
  ///         The values.
  const T* values() const;

  /// Return array of values
  ///
  /// return T
  ///         The values.
  T* values();

  /// Return value at given mesh entity
  ///
  /// @param entity (_MeshEntity_)
  ///         The mesh entity.
  ///
  /// return    T
  ///         The value at the given entity.
  T& operator[](const MeshEntity& entity);

  /// Return value at given mesh entity (const version)
  ///
  /// @param entity (_MeshEntity_)
  ///         The mesh entity.
  ///
  /// @return T
  ///         The value at the given entity.
  const T& operator[](const MeshEntity& entity) const;

  /// Return value at given index
  ///
  /// @param index (std::size_t)
  ///         The index.
  ///
  /// @return T
  ///         The value at the given index.
  T& operator[](std::size_t index);

  /// Return value at given index  (const version)
  ///
  /// @param index (std::size_t)
  ///         The index.
  ///
  /// @return T
  ///         The value at the given index.
  const T& operator[](std::size_t index) const;

  /// Set all values to given value
  /// @param value (T)
  const MeshFunction<T>& operator=(const T& value);

  /// Set value at given index
  ///
  /// @param index (std::size_t)
  ///         The index.
  /// @param value (T)
  ///         The value.
  void set_value(std::size_t index, const T& value);

  /// Compatibility function for use in SubDomains
  void set_value(std::size_t index, const T& value, const Mesh& mesh)
  {
    set_value(index, value);
  }

  /// Set values
  ///
  /// @param values (std::vector<T>)
  ///         The values.
  void set_values(const std::vector<T>& values);

  /// Get indices where meshfunction is equal to given value
  ///
  /// *Arguments*
  ///     value (T)
  ///         The value.
  /// *Returns*
  ///     std::vector<T>
  ///         The indices.
  std::vector<std::size_t> where_equal(T value);

  /// Return informal string representation (pretty-print)
  ///
  /// @param verbose (bool)
  ///         Flag to turn on additional output.
  ///
  /// @return std::string
  ///         An informal representation.
  std::string str(bool verbose) const;

private:
  // Values at the set of mesh entities. We don't use a
  // std::vector<T> here because it has trouble with bool, which C++
  // specialises.
  boost::container::vector<T> _values;

  // The mesh
  std::shared_ptr<const Mesh> _mesh;

  // Topological dimension
  std::size_t _dim;
};

template <>
std::string MeshFunction<double>::str(bool verbose) const;
template <>
std::string MeshFunction<std::size_t>::str(bool verbose) const;

//---------------------------------------------------------------------------
// Implementation of MeshFunction
//---------------------------------------------------------------------------
template <typename T>
MeshFunction<T>::MeshFunction(std::shared_ptr<const Mesh> mesh, std::size_t dim,
                              const T& value)
    : _mesh(mesh), _dim(dim)
{
  assert(mesh);
  mesh->init(dim);
  _values.resize(mesh->num_entities(dim), value);
}
//---------------------------------------------------------------------------
template <typename T>
MeshFunction<T>::MeshFunction(std::shared_ptr<const Mesh> mesh,
                              const MeshValueCollection<T>& value_collection)
    : common::Variable("f", "unnamed MeshFunction"), _mesh(mesh),
      _dim(value_collection.dim())
{
  *this = value_collection;
}
//---------------------------------------------------------------------------
template <typename T>
MeshFunction<T>& MeshFunction<T>::
operator=(const MeshValueCollection<T>& mesh_value_collection)
{
  _dim = mesh_value_collection.dim();
  dolfin_assert(_mesh);
  _mesh->init(_dim);
  _values.resize(_mesh->topology().size(_dim));

  // Get mesh connectivity D --> d
  const std::size_t d = _dim;
  const std::size_t D = _mesh->topology().dim();
  dolfin_assert(d <= D);

  // Generate connectivity if it does not exist
  _mesh->init(D, d);
  const MeshConnectivity& connectivity = _mesh->topology()(D, d);
  dolfin_assert(!connectivity.empty());

  // Set MeshFunction with default value
  // set_all(std::numeric_limits<T>::max());

  // Iterate over all values
  std::unordered_set<std::size_t> entities_values_set;
  typename std::map<std::pair<std::size_t, std::size_t>, T>::const_iterator it;
  const std::map<std::pair<std::size_t, std::size_t>, T>& values
      = mesh_value_collection.values();
  for (it = values.begin(); it != values.end(); ++it)
  {
    // Get value collection entry data
    const std::size_t cell_index = it->first.first;
    const std::size_t local_entity = it->first.second;
    const T value = it->second;

    std::size_t entity_index = 0;
    if (d != D)
    {
      // Get global (local to to process) entity index
      dolfin_assert(cell_index < _mesh->num_cells());
      entity_index = connectivity(cell_index)[local_entity];
    }
    else
    {
      entity_index = cell_index;
      dolfin_assert(local_entity == 0);
    }

    // Set value for entity
    dolfin_assert(entity_index < _values.size());
    _values[entity_index] = value;

    // Add entity index to set (used to check that all values are set)
    entities_values_set.insert(entity_index);
  }

  // Check that all values have been set, if not issue a debug message
  if (entities_values_set.size() != _values.size())
    dolfin_debug(
        "Mesh value collection does not contain all values for all entities");

  return *this;
}
//---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<const Mesh> MeshFunction<T>::mesh() const
{
  dolfin_assert(_mesh);
  return _mesh;
}
//---------------------------------------------------------------------------
template <typename T>
std::size_t MeshFunction<T>::dim() const
{
  return _dim;
}
//---------------------------------------------------------------------------
template <typename T>
std::size_t MeshFunction<T>::size() const
{
  return _values.size();
}
//---------------------------------------------------------------------------
template <typename T>
const T* MeshFunction<T>::values() const
{
  return _values.data();
}
//---------------------------------------------------------------------------
template <typename T>
T* MeshFunction<T>::values()
{
  return _values.data();
}
//---------------------------------------------------------------------------
template <typename T>
T& MeshFunction<T>::operator[](const MeshEntity& entity)
{
  dolfin_assert(&entity.mesh() == _mesh.get());
  dolfin_assert(entity.dim() == _dim);
  dolfin_assert((std::uint32_t)entity.index() < _values.size());
  return _values[entity.index()];
}
//---------------------------------------------------------------------------
template <typename T>
const T& MeshFunction<T>::operator[](const MeshEntity& entity) const
{
  dolfin_assert(&entity.mesh() == _mesh.get());
  dolfin_assert(entity.dim() == _dim);
  dolfin_assert((std::uint32_t)entity.index() < _values.size());
  return _values[entity.index()];
}
//---------------------------------------------------------------------------
template <typename T>
T& MeshFunction<T>::operator[](std::size_t index)
{
  dolfin_assert(index < _values.size());
  return _values[index];
}
//---------------------------------------------------------------------------
template <typename T>
const T& MeshFunction<T>::operator[](std::size_t index) const
{
  dolfin_assert(index < _values.size());
  return _values[index];
}
//---------------------------------------------------------------------------
template <typename T>
const MeshFunction<T>& MeshFunction<T>::operator=(const T& value)
{
  _values = value;
  // std::fill(_values.begin(), _values.end(), value);
  // set_all(value);
  return *this;
}
//---------------------------------------------------------------------------
template <typename T>
void MeshFunction<T>::set_value(std::size_t index, const T& value)
{
  dolfin_assert(index < _values.size());
  _values[index] = value;
}
//---------------------------------------------------------------------------
template <typename T>
void MeshFunction<T>::set_values(const std::vector<T>& values)
{
  dolfin_assert(_values.size() == values.size());
  std::copy(values.begin(), values.end(), _values.begin());
}
//---------------------------------------------------------------------------
template <typename T>
std::vector<std::size_t> MeshFunction<T>::where_equal(T value)
{
  std::size_t n = std::count(_values.begin(), _values.end(), value);
  std::vector<std::size_t> indices;
  indices.reserve(n);
  for (std::size_t i = 0; i < size(); ++i)
  {
    if (_values[i] == value)
      indices.push_back(i);
  }
  return indices;
}
//---------------------------------------------------------------------------
template <typename T>
std::string MeshFunction<T>::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    log::warning(
        "Verbose output of MeshFunctions must be implemented manually.");
  }
  else
  {
    s << "<MeshFunction of topological dimension " << dim() << " containing "
      << size() << " values>";
  }

  return s.str();
}
//---------------------------------------------------------------------------
}
}