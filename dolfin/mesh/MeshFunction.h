// Copyright (C) 2006-2009 Anders Logg
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
// Modified by Johan Hoffman, 2007.
// Modified by Garth N. Wells, 2010-2013
//
// First added:  2006-05-22
// Last changed: 2013-05-10

#ifndef __MESH_FUNCTION_H
#define __MESH_FUNCTION_H

#include <map>
#include <vector>

#include <memory>
#include <unordered_set>
#include <dolfin/common/Hierarchical.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>
#include <dolfin/io/File.h>
#include "LocalMeshValueCollection.h"
#include "MeshDomains.h"
#include "MeshEntity.h"
#include "Mesh.h"
#include "MeshConnectivity.h"

namespace dolfin
{

  class MeshEntity;

  /// A MeshFunction is a function that can be evaluated at a set of
  /// mesh entities. A MeshFunction is discrete and is only defined at
  /// the set of mesh entities of a fixed topological dimension.  A
  /// MeshFunction may for example be used to store a global numbering
  /// scheme for the entities of a (parallel) mesh, marking sub
  /// domains or boolean markers for mesh refinement.

  template <typename T> class MeshFunction : public Variable,
    public Hierarchical<MeshFunction<T>>
  {
  public:

    /// Create empty mesh function
    MeshFunction();

    /// Create empty mesh function on given mesh
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh function on.
    explicit MeshFunction(std::shared_ptr<const Mesh> mesh);

    /// Create mesh function of given dimension on given mesh
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh function on.
    ///     dim (std::size_t)
    ///         The mesh entity dimension for the mesh function.
    MeshFunction(std::shared_ptr<const Mesh> mesh, std::size_t dim);

    /// Create mesh of given dimension on given mesh and initialize
    /// to a value
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh function on.
    ///     dim (std::size_t)
    ///         The mesh entity dimension.
    ///     value (T)
    ///         The value.
    MeshFunction(std::shared_ptr<const Mesh> mesh, std::size_t dim,
                 const T& value);

    /// Create function from data file (shared_ptr version)
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh function on.
    ///     filename (std::string)
    ///         The filename to create mesh function from.
    MeshFunction(std::shared_ptr<const Mesh> mesh,
                 const std::string filename);

    /// Create function from a MeshValueCollecion (shared_ptr version)
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh function on.
    ///     value_collection (_MeshValueCollection_ <T>)
    ///         The mesh value collection for the mesh function data.
    MeshFunction(std::shared_ptr<const Mesh> mesh,
                 const MeshValueCollection<T>& value_collection);

    /// Create function from MeshDomains
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh function on.
    ///     dim (std::size_t)
    ///         The dimension of the MeshFunction
    ///     domains (_MeshDomains)
    ///         The domains from which to extract the domain markers
    MeshFunction(std::shared_ptr<const Mesh> mesh,
                 std::size_t dim, const MeshDomains& domains);

    /// Copy constructor
    ///
    /// *Arguments*
    ///     f (_MeshFunction_ <T>)
    ///         The object to be copied.
    MeshFunction(const MeshFunction<T>& f);

    /// Destructor
    ~MeshFunction() {}

    /// Assign mesh function to other mesh function
    /// Assignment operator
    ///
    /// *Arguments*
    ///     f (_MeshFunction_ <T>)
    ///         A _MeshFunction_ object to assign to another MeshFunction.
    MeshFunction<T>& operator= (const MeshFunction<T>& f);

    /// Assignment operator
    ///
    /// *Arguments*
    ///     mesh (_MeshValueCollection_)
    ///         A _MeshValueCollection_ object used to construct a MeshFunction.
    MeshFunction<T>& operator=(const MeshValueCollection<T>& mesh);

    /// Return mesh associated with mesh function
    ///
    /// *Returns*
    ///     _Mesh_
    ///         The mesh.
    std::shared_ptr<const Mesh> mesh() const;

    /// Return topological dimension
    ///
    /// *Returns*
    ///     std::size_t
    ///         The dimension.
    std::size_t dim() const;

    /// Return true if empty
    ///
    /// *Returns*
    ///     bool
    ///         True if empty.
    bool empty() const;

    /// Return size (number of entities)
    ///
    /// *Returns*
    ///     std::size_t
    ///         The size.
    std::size_t size() const;

    /// Return array of values (const. version)
    ///
    /// *Returns*
    ///     T
    ///         The values.
    const T* values() const;

    /// Return array of values
    ///
    /// *Returns*
    ///     T
    ///         The values.
    T* values();

    /// Return value at given mesh entity
    ///
    /// *Arguments*
    ///     entity (_MeshEntity_)
    ///         The mesh entity.
    ///
    /// *Returns*
    ///     T
    ///         The value at the given entity.
    T& operator[] (const MeshEntity& entity);

    /// Return value at given mesh entity (const version)
    ///
    /// *Arguments*
    ///     entity (_MeshEntity_)
    ///         The mesh entity.
    ///
    /// *Returns*
    ///     T
    ///         The value at the given entity.
    const T& operator[] (const MeshEntity& entity) const;

    /// Return value at given index
    ///
    /// *Arguments*
    ///     index (std::size_t)
    ///         The index.
    ///
    /// *Returns*
    ///     T
    ///         The value at the given index.
    T& operator[] (std::size_t index);

    /// Return value at given index  (const version)
    ///
    /// *Arguments*
    ///     index (std::size_t)
    ///         The index.
    ///
    /// *Returns*
    ///     T
    ///         The value at the given index.
    const T& operator[] (std::size_t index) const;

    /// Set all values to given value
    const MeshFunction<T>& operator= (const T& value);

    /// Initialize mesh function for given topological dimension
    ///
    /// *Arguments*
    ///     dim (std::size_t)
    ///         The dimension.
    void init(std::size_t dim);

    /// Initialize mesh function for given topological dimension of
    /// given size
    ///
    /// *Arguments*
    ///     dim (std::size_t)
    ///         The dimension.
    ///     size (std::size_t)
    ///         The size.
    void init(std::size_t dim, std::size_t size);

    /// Initialize mesh function for given topological dimension
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///     dim (std::size_t)
    ///         The dimension.
    void init(std::shared_ptr<const Mesh> mesh, std::size_t dim);

    /// Initialize mesh function for given topological dimension of
    /// given size (shared_ptr version)
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///     dim (std::size_t)
    ///         The dimension.
    ///     size (std::size_t)
    ///         The size.
    void init(std::shared_ptr<const Mesh> mesh, std::size_t dim,
              std::size_t size);

    /// Set value at given index
    ///
    /// *Arguments*
    ///     index (std::size_t)
    ///         The index.
    ///     value (T)
    ///         The value.
    void set_value(std::size_t index, const T& value);

    /// Compatibility function for use in SubDomains
    void set_value(std::size_t index, const T& value, const Mesh& mesh)
    { set_value(index, value); }

    /// Set values
    ///
    /// *Arguments*
    ///     values (std::vector<T>)
    ///         The values.
    void set_values(const std::vector<T>& values);

    /// Set all values to given value
    ///
    /// *Arguments*
    ///     value (T)
    ///         The value to set all values to.
    void set_all(const T& value);

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
    /// *Arguments*
    ///     verbose (bool)
    ///         Flag to turn on additional output.
    ///
    /// *Returns*
    ///     std::string
    ///         An informal representation.
    std::string str(bool verbose) const;

  private:

    // Values at the set of mesh entities. We don't use a
    // std::vector<T> here because it has trouble with bool, which C++
    // specialises.
    std::unique_ptr<T[]> _values;

    // The mesh
    std::shared_ptr<const Mesh> _mesh;

    // Topological dimension
    std::size_t _dim;

    // Number of mesh entities
    std::size_t _size;
  };

  template<> std::string MeshFunction<double>::str(bool verbose) const;
  template<> std::string MeshFunction<std::size_t>::str(bool verbose) const;

  //---------------------------------------------------------------------------
  // Implementation of MeshFunction
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction() : MeshFunction(nullptr)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction(std::shared_ptr<const Mesh> mesh)
    : Variable("f", "unnamed MeshFunction"),
      Hierarchical<MeshFunction<T>>(*this), _mesh(mesh), _dim(0), _size(0)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction(std::shared_ptr<const Mesh> mesh,
                                std::size_t dim)
    : Variable("f", "unnamed MeshFunction"),
      Hierarchical<MeshFunction<T>>(*this), _mesh(mesh), _dim(0), _size(0)
  {
    init(dim);
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction(std::shared_ptr<const Mesh> mesh,
                                std::size_t dim, const T& value)
    : MeshFunction(mesh, dim)

  {
    set_all(value);
  }
  //---------------------------------------------------------------------------
  template <typename T>
    MeshFunction<T>::MeshFunction(std::shared_ptr<const Mesh> mesh,
                                  const std::string filename)
    : Variable("f", "unnamed MeshFunction"),
    Hierarchical<MeshFunction<T>>(*this), _mesh(mesh), _dim(0), _size(0)
  {
    File file(mesh->mpi_comm(), filename);
    file >> *this;
  }
  //---------------------------------------------------------------------------
  template <typename T>
    MeshFunction<T>::MeshFunction(std::shared_ptr<const Mesh> mesh,
                                  const MeshValueCollection<T>& value_collection)
    : Variable("f", "unnamed MeshFunction"),
      Hierarchical<MeshFunction<T>>(*this), _mesh(mesh),
      _dim(value_collection.dim()), _size(0)
  {
    *this = value_collection;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction(std::shared_ptr<const Mesh> mesh,
                                std::size_t dim, const MeshDomains& domains)
    : Variable("f", "unnamed MeshFunction"),
      Hierarchical<MeshFunction<T>>(*this), _mesh(mesh), _dim(0), _size(0)
  {
    dolfin_assert(_mesh);

    // Initialise MeshFunction
    init(dim);

    // Initialise mesh
    mesh->init(dim);

    // Set MeshFunction with default value
    set_all(std::numeric_limits<T>::max());

    // Get mesh dimension
    const std::size_t D = _mesh->topology().dim();
    dolfin_assert(dim <= D);

    // Get domain data
    const std::map<std::size_t, std::size_t>& data = domains.markers(dim);

    // Iterate over all values and copy into MeshFunctions
    std::map<std::size_t, std::size_t>::const_iterator it;
    for (it = data.begin(); it != data.end(); ++it)
    {
      // Get value collection entry data
      const std::size_t entity_index = it->first;
      const T value = it->second;

      dolfin_assert(entity_index < _size);
      _values[entity_index] = value;
    }
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction(const MeshFunction<T>& f) :
    Variable("f", "unnamed MeshFunction"),
    Hierarchical<MeshFunction<T>>(*this), _dim(0), _size(0)
  {
    *this = f;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>& MeshFunction<T>::operator= (const MeshFunction<T>& f)
  {
    if (_size != f._size)
      _values.reset(new T[f._size]);
    _mesh = f._mesh;
    _dim  = f._dim;
    _size = f._size;
    std::copy(f._values.get(), f._values.get() + _size, _values.get());

    Hierarchical<MeshFunction<T>>::operator=(f);

    return *this;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>& MeshFunction<T>::operator=(const MeshValueCollection<T>& mesh_value_collection)
  {
    _dim = mesh_value_collection.dim();
    init(_dim);
    dolfin_assert(_mesh);

    // Get mesh connectivity D --> d
    const std::size_t d = _dim;
    const std::size_t D = _mesh->topology().dim();
    dolfin_assert(d <= D);

    // Generate connectivity if it does not exist
    _mesh->init(D, d);
    const MeshConnectivity& connectivity = _mesh->topology()(D, d);
    dolfin_assert(!connectivity.empty());

    // Set MeshFunction with default value
    set_all(std::numeric_limits<T>::max());

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
      dolfin_assert(entity_index < _size);
      _values[entity_index] = value;

      // Add entity index to set (used to check that all values are set)
      entities_values_set.insert(entity_index);
    }

    // Check that all values have been set, if not issue a debug message
    if (entities_values_set.size() != _size)
      dolfin_debug("Mesh value collection does not contain all values for all entities");

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
    bool MeshFunction<T>::empty() const
  {
    return _size == 0;
  }
  //---------------------------------------------------------------------------
  template <typename T>
    std::size_t MeshFunction<T>::size() const
  {
    return _size;
  }
  //---------------------------------------------------------------------------
  template <typename T>
    const T* MeshFunction<T>::values() const
  {
    return _values.get();
  }
  //---------------------------------------------------------------------------
  template <typename T>
    T* MeshFunction<T>::values()
  {
    return _values.get();
  }
  //---------------------------------------------------------------------------
  template <typename T>
    T& MeshFunction<T>::operator[] (const MeshEntity& entity)
  {
    dolfin_assert(_values);
    dolfin_assert(&entity.mesh() == _mesh.get());
    dolfin_assert(entity.dim() == _dim);
    dolfin_assert(entity.index() < _size);
    return _values[entity.index()];
  }
  //---------------------------------------------------------------------------
  template <typename T>
    const T& MeshFunction<T>::operator[] (const MeshEntity& entity) const
  {
    dolfin_assert(_values);
    dolfin_assert(&entity.mesh() == _mesh.get());
    dolfin_assert(entity.dim() == _dim);
    dolfin_assert(entity.index() < _size);
    return _values[entity.index()];
  }
  //---------------------------------------------------------------------------
  template <typename T>
    T& MeshFunction<T>::operator[] (std::size_t index)
  {
    dolfin_assert(_values);
    dolfin_assert(index < _size);
    return _values[index];
  }
  //---------------------------------------------------------------------------
  template <typename T>
    const T& MeshFunction<T>::operator[] (std::size_t index) const
  {
    dolfin_assert(_values);
    dolfin_assert(index < _size);
    return _values[index];
  }
  //---------------------------------------------------------------------------
  template <typename T>
    const MeshFunction<T>& MeshFunction<T>::operator= (const T& value)
  {
    set_all(value);
    //Hierarchical<MeshFunction<T>>::operator=(value);
    return *this;
  }
  //---------------------------------------------------------------------------
  template <typename T>
    void MeshFunction<T>::init(std::size_t dim)
  {
    if (!_mesh)
    {
      dolfin_error("MeshFunction.h",
                   "initialize mesh function",
                   "Mesh has not been specified for mesh function");

    }
    _mesh->init(dim);
    init(_mesh, dim, _mesh->size(dim));
  }
  //---------------------------------------------------------------------------
  template <typename T>
    void MeshFunction<T>::init(std::size_t dim, std::size_t size)
  {
    if (!_mesh)
    {
      dolfin_error("MeshFunction.h",
                   "initialize mesh function",
                   "Mesh has not been specified for mesh function");
    }
    _mesh->init(dim);
    init(_mesh, dim, size);
  }
  //---------------------------------------------------------------------------
  template <typename T>
    void MeshFunction<T>::init(std::shared_ptr<const Mesh> mesh,
                               std::size_t dim)
  {
    dolfin_assert(mesh);
    mesh->init(dim);
    init(mesh, dim, mesh->size(dim));
  }
  //---------------------------------------------------------------------------
  template <typename T>
    void MeshFunction<T>::init(std::shared_ptr<const Mesh> mesh,
                               std::size_t dim, std::size_t size)
  {
    dolfin_assert(mesh);

    // Initialize mesh for entities of given dimension
    mesh->init(dim);
    dolfin_assert(mesh->size(dim) == size);

    // Initialize data
    if (_size != size)
      _values.reset(new T[size]);
    _mesh = mesh;
    _dim = dim;
    _size = size;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshFunction<T>::set_value(std::size_t index, const T& value)
  {
    dolfin_assert(_values);
    dolfin_assert(index < _size);
    _values[index] = value;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshFunction<T>::set_values(const std::vector<T>& values)
  {
    dolfin_assert(_values);
    dolfin_assert(_size == values.size());
    std::copy(values.begin(), values.end(), _values.get());
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshFunction<T>::set_all(const T& value)
  {
    dolfin_assert(_values);
    std::fill(_values.get(), _values.get() + _size, value);
  }
  //---------------------------------------------------------------------------
  template <typename T>
  std::vector<std::size_t> MeshFunction<T>::where_equal(T value)
  {
    dolfin_assert(_values);
    std::size_t n = std::count(_values.get(), _values.get() + _size, value);
    std::vector<std::size_t> indices;
    indices.reserve(n);
    for (int i = 0; i < size(); ++i)
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
      warning("Verbose output of MeshFunctions must be implemented manually.");

      // This has been disabled as it severely restricts the ease with which
      // templated MeshFunctions can be used, e.g. it is not possible to
      // template over std::vector.

      //for (std::size_t i = 0; i < _size; i++)
      //  s << "  (" << _dim << ", " << i << "): " << _values[i] << std::endl;
    }
    else
    {
      s << "<MeshFunction of topological dimension " << dim()
        << " containing " << size() << " values>";
    }

    return s.str();
  }
  //---------------------------------------------------------------------------

}

#endif
