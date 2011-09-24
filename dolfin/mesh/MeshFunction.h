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
// Modified by Garth N. Wells, 2010.
//
// First added:  2006-05-22
// Last changed: 2011-03-12

#ifndef __MESH_FUNCTION_H
#define __MESH_FUNCTION_H

#include <vector>
#include <boost/unordered_set.hpp>
#include <dolfin/common/Hierarchical.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>
#include <dolfin/io/File.h>
#include "LocalMeshValueCollection.h"
#include "MeshEntity.h"
#include "Mesh.h"
#include "MeshConnectivity.h"
#include "MeshPartitioning.h"

namespace dolfin
{

  class MeshEntity;

  /// A MeshFunction is a function that can be evaluated at a set of
  /// mesh entities. A MeshFunction is discrete and is only defined
  /// at the set of mesh entities of a fixed topological dimension.
  /// A MeshFunction may for example be used to store a global
  /// numbering scheme for the entities of a (parallel) mesh, marking
  /// sub domains or boolean markers for mesh refinement.

  template <typename T> class MeshFunction : public Variable,
    public Hierarchical<MeshFunction<T> >
  {
  public:

    /// Create empty mesh function
    MeshFunction();

    /// Create empty mesh function on given mesh
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh function on.
    explicit MeshFunction(const Mesh& mesh);

    /// Create mesh function of given dimension on given mesh
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh function on.
    ///     dim (uint)
    ///         The mesh entity dimension for the mesh function.
    MeshFunction(const Mesh& mesh, uint dim);

    /// Create mesh of given dimension on given mesh and initialize
    /// to a value
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh function on.
    ///     dim (uint)
    ///         The mesh entity dimension.
    ///     value (T)
    ///         The value.
    MeshFunction(const Mesh& mesh, uint dim, const T& value);

    /// Create function from data file
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh function on.
    ///     filename (std::string)
    ///         The filename to create mesh function from.
    MeshFunction(const Mesh& mesh, const std::string filename);

    /// Create function from a MeshValueCollecion
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh to create mesh function on.
    ///     value_collection (_MeshValueCollection_ <T>)
    ///         The mesh value collection for the mesh function data.
    MeshFunction(const Mesh& mesh,
                 const MeshValueCollection<T>& value_collection);

    /// Copy constructor
    ///
    /// *Arguments*
    ///     f (_MeshFunction_ <T>)
    ///         The object to be copied.
    MeshFunction(const MeshFunction<T>& f);

    /// Destructor
    ~MeshFunction()
    { delete [] _values; }

    /// Assignment operator
    ///
    /// *Arguments*
    ///     mesh (_MeshValueCollection_)
    ///         A _MeshValueCollection_ object used to construct a MeshFunction.
    const MeshFunction<T>& operator=(const MeshValueCollection<T>& mesh);

    /// Return mesh associated with mesh function
    ///
    /// *Returns*
    ///     _Mesh_
    ///         The mesh.
    const Mesh& mesh() const;

    /// Return topological dimension
    ///
    /// *Returns*
    ///     uint
    ///         The dimension.
    uint dim() const;

    /// Return size (number of entities)
    ///
    /// *Returns*
    ///     uint
    ///         The size.
    uint size() const;

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
    ///     index (uint)
    ///         The index.
    ///
    /// *Returns*
    ///     T
    ///         The value at the given index.
    T& operator[] (uint index);

    /// Return value at given index  (const version)
    ///
    /// *Arguments*
    ///     index (uint)
    ///         The index.
    ///
    /// *Returns*
    ///     T
    ///         The value at the given index.
    const T& operator[] (uint index) const;

    /// Assign mesh function to other mesh function
    const MeshFunction<T>& operator= (const MeshFunction<T>& f);

    /// Set all values to given value
    const MeshFunction<T>& operator= (const T& value);

    /// Initialize mesh function for given topological dimension
    ///
    /// *Arguments*
    ///     dim (uint)
    ///         The dimension.
    void init(uint dim);

    /// Initialize mesh function for given topological dimension of
    /// given size
    ///
    /// *Arguments*
    ///     dim (uint)
    ///         The dimension.
    ///     size (uint)
    ///         The size.
    void init(uint dim, uint size);

    /// Initialize mesh function for given topological dimension
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///     dim (uint)
    ///         The dimension.
    void init(const Mesh& mesh, uint dim);

    /// Initialize mesh function for given topological dimension of
    /// given size
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///     dim (uint)
    ///         The dimension.
    ///     size (uint)
    ///         The size.
    void init(const Mesh& mesh, uint dim, uint size);

    /// Set value at given index
    ///
    /// *Arguments*
    ///     index (uint)
    ///         The index.
    ///     value (T)
    ///         The value.
    void set_value(uint index, T& value);

    /// Compatibility function for use in SubDomains
    void set_value(uint index, T& value, const Mesh& mesh)
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

    /// Values at the set of mesh entities
    T* _values;

    /// The mesh
    const Mesh* _mesh;

    /// Topological dimension
    uint _dim;

    /// Number of mesh entities
    uint _size;
  };

  template<> std::string MeshFunction<double>::str(bool verbose) const;
  template<> std::string MeshFunction<uint>::str(bool verbose) const;

  //---------------------------------------------------------------------------
  // Implementation of MeshFunction
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction() : Variable("f", "unnamed MeshFunction"),
    Hierarchical<MeshFunction<T> >(*this),
    _values(0), _mesh(0), _dim(0), _size(0)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction(const Mesh& mesh) :
      Variable("f", "unnamed MeshFunction"),
      Hierarchical<MeshFunction<T> >(*this),
      _values(0), _mesh(&mesh), _dim(0), _size(0)
  {
    // Do nothing
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction(const Mesh& mesh, uint dim) :
      Variable("f", "unnamed MeshFunction"),
      Hierarchical<MeshFunction<T> >(*this),
      _values(0), _mesh(&mesh), _dim(0), _size(0)
  {
    init(dim);
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction(const Mesh& mesh, uint dim, const T& value) :
      Variable("f", "unnamed MeshFunction"),
      Hierarchical<MeshFunction<T> >(*this),
      _values(0), _mesh(&mesh), _dim(0), _size(0)
  {
    init(dim);
    set_all(value);
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction(const Mesh& mesh, const std::string filename) :
      Variable("f", "unnamed MeshFunction"),
      Hierarchical<MeshFunction<T> >(*this),
      _values(0), _mesh(&mesh), _dim(0), _size(0)
  {
    File file(filename);
    file >> *this;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction(const Mesh& mesh,
      const MeshValueCollection<T>& value_collection) :
      Variable("f", "unnamed MeshFunction"),
      Hierarchical<MeshFunction<T> >(*this),
      _values(0), _mesh(&mesh), _dim(value_collection.dim()), _size(0)
  {
    *this = value_collection;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  MeshFunction<T>::MeshFunction(const MeshFunction<T>& f) :
      Variable("f", "unnamed MeshFunction"),
      Hierarchical<MeshFunction<T> >(*this),
      _values(0), _mesh(0), _dim(0), _size(0)
  {
    *this = f;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  const MeshFunction<T>& MeshFunction<T>::operator=(const MeshValueCollection<T>& mesh_value_collection)
  {
    _dim = mesh_value_collection.dim();
    init(_dim);
    assert(_mesh);

    // Get mesh connectivity D --> d
    const uint d = _dim;
    const uint D = _mesh->topology().dim();
    assert(d <= D);
    const MeshConnectivity& connectivity = _mesh->topology()(D, d);
    assert(connectivity.size() > 0);

    // Iterate over all values
    boost::unordered_set<uint> entities_values_set;
    typename std::map<std::pair<uint, uint>, T>::const_iterator it;
    const std::map<std::pair<uint, uint>, T>& values = mesh_value_collection.values();
    for (it = values.begin(); it != values.end(); ++it)
    {
      // Get value collection entry data
      const uint cell_index = it->first.first;
      const uint local_entity = it->first.second;
      const T value = it->second;

      uint entity_index = 0;
      if (d != D)
      {
        // Get global (local to to process) entity index
        assert(cell_index < _mesh->num_cells());
        entity_index = connectivity(cell_index)[local_entity];

      }
      else
      {
        entity_index = cell_index;
        assert(local_entity == 0);
      }

      // Set value for entity
      assert(entity_index < _size);
      _values[entity_index] = value;

      // Add entity index to set (used to check that all values are set)
      entities_values_set.insert(entity_index);
    }

    // Check that all values have been set
    if (entities_values_set.size() != _size)
      error("MeshFunction<T>::operator=: MeshValueCollection does not contain all values for all entities. Cannot construct MeshFunction.");

    return *this;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  const Mesh& MeshFunction<T>::mesh() const
  {
    assert(_mesh);
    return *_mesh;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  uint MeshFunction<T>::dim() const
  {
    return _dim;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  uint MeshFunction<T>::size() const
  {
    return _size;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  const T* MeshFunction<T>::values() const
  {
    return _values;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  T* MeshFunction<T>::values()
  {
    return _values;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  T& MeshFunction<T>::operator[] (const MeshEntity& entity)
  {
    assert(_values);
    assert(&entity.mesh() == _mesh);
    assert(entity.dim() == _dim);
    assert(entity.index() < _size);
    return _values[entity.index()];
  }
  //---------------------------------------------------------------------------
  template <typename T>
  const T& MeshFunction<T>::operator[] (const MeshEntity& entity) const
  {
    assert(_values);
    assert(&entity.mesh() == _mesh);
    assert(entity.dim() == _dim);
    assert(entity.index() < _size);
    return _values[entity.index()];
  }
  //---------------------------------------------------------------------------
  template <typename T>
  T& MeshFunction<T>::operator[] (uint index)
  {
    assert(_values);
    assert(index < _size);
    return _values[index];
  }
  //---------------------------------------------------------------------------
  template <typename T>
  const T& MeshFunction<T>::operator[] (uint index) const
  {
    assert(_values);
    assert(index < _size);
    return _values[index];
  }
  //---------------------------------------------------------------------------
  template <typename T>
  const MeshFunction<T>& MeshFunction<T>::operator= (const MeshFunction<T>& f)
  {
    _mesh = f._mesh;
    _dim = f._dim;
    _size = f._size;
    delete [] _values;
    _values = new T[_size];
    for (uint i = 0; i < _size; i++)
      _values[i] = f._values[i];

    Hierarchical<MeshFunction<T> >::operator=(f);

    return *this;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  const MeshFunction<T>& MeshFunction<T>::operator= (const T& value)
  {
    set_all(value);
    //Hierarchical<MeshFunction<T> >::operator=(value);
    return *this;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshFunction<T>::init(uint dim)
  {
    if (!_mesh)
      error("Mesh has not been specified, unable to initialize mesh function.");
    _mesh->init(dim);
    init(*_mesh, dim, _mesh->size(dim));
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshFunction<T>::init(uint dim, uint size)
  {
    if (!_mesh)
      error("Mesh has not been specified, unable to initialize mesh function.");
    _mesh->init(dim);
    init(*_mesh, dim, size);
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshFunction<T>::init(const Mesh& mesh, uint dim)
  {
    mesh.init(dim);
    init(mesh, dim, mesh.size(dim));
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshFunction<T>::init(const Mesh& mesh, uint dim, uint size)
  {
    // Initialize mesh for entities of given dimension
    mesh.init(dim);
    assert(mesh.size(dim) == size);

    // Initialize data
    _mesh = &mesh;
    _dim = dim;
    _size = size;
    delete [] _values;
    _values = new T[size];
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshFunction<T>::set_value(uint index, T& value)
  {
    assert(_values);
    assert(index < _size);
    _values[index] = value;
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshFunction<T>::set_values(const std::vector<T>& values)
  {
    assert(_values);
    assert(_size == values.size());
    for (uint i = 0; i < _size; i++)
      _values[i] = values[i];
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void MeshFunction<T>::set_all(const T& value)
  {
    assert(_values);
    for (uint i = 0; i < _size; i++)
      _values[i] = value;
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

      //for (uint i = 0; i < _size; i++)
      //  s << "  (" << _dim << ", " << i << "): " << _values[i] << std::endl;
    }
    else
      s << "<MeshFunction of topological dimension " << dim()
        << " containing " << size() << " values>";

    return s.str();
  }
  //---------------------------------------------------------------------------

}

#endif
