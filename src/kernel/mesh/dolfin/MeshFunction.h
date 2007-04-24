// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman 2007.
//
// First added:  2006-05-22
// Last changed: 2007-04-24

#ifndef __MESH_FUNCTION_H
#define __MESH_FUNCTION_H

#include <dolfin/constants.h>
#include <dolfin/MeshEntity.h>

namespace dolfin
{

  /// A MeshFunction is a function that can be evaluated at a set of
  /// mesh entities. A MeshFunction is discrete and is only defined
  /// at the set of mesh entities of a fixed topological dimension.
  /// A MeshFunction may for example be used to store a global
  /// numbering scheme for the entities of a (parallel) mesh, marking
  /// sub domains or boolean markers for mesh refinement.

  template <class T> class MeshFunction
  {
  public:

    /// Create empty mesh function
    MeshFunction() : _values(0), _mesh(0), _dim(0), _size(0) {}

    /// Create empty mesh function on given mesh
    MeshFunction(Mesh& mesh) : _values(0), _mesh(&mesh), _dim(0), _size(0) {}

    /// Create mesh function on given mesh of given dimension
    MeshFunction(Mesh& mesh, uint dim) : _values(0), _mesh(&mesh), _dim(0), _size(0)
    {
      init(dim);
    }

    /// Destructor
    ~MeshFunction()
    {
      if ( _values )
	delete [] _values;
    }

    /// Return mesh associated with mesh function
    inline Mesh& mesh() { dolfin_assert(_mesh); return *_mesh; }

    /// Return topological dimension
    inline uint dim() const { return _dim; }

    /// Return size (number of entities)
    inline uint size() const { return _size; }

    /// Return array of values
    inline const T* values() const { return _values; }

    /// Return value at given entity
    inline T& operator() (MeshEntity& entity)
    {
      dolfin_assert(_values);
      dolfin_assert(&entity.mesh() == _mesh);
      dolfin_assert(entity.dim() == _dim);
      dolfin_assert(entity.index() < _size);
      return _values[entity.index()];
    }

    /// Return value at given entity
    inline const T& operator() (MeshEntity& entity) const
    {
      dolfin_assert(_values);
      dolfin_assert(&entity.mesh() == _mesh);
      dolfin_assert(entity.dim() == _dim);
      dolfin_assert(entity.index() < _size);
      return _values[entity.index()];
    }

    /// Set all values to given value
    const MeshFunction<T>& operator= (const T& value)
    {
      dolfin_assert(_values);
      for (uint i = 0; i < _size; i++)
        _values[i] = value;
      return *this;
    }

    /// Initialize mesh function for given topological dimension
    void init(uint dim)
    {
      if ( !_mesh )
        dolfin_error("Mesh has not been specified, unable to initialize mesh function.");
      init(*_mesh, dim, _mesh->size(dim));
    }

    /// Initialize mesh function for given topological dimension of given size
    void init(uint dim, uint size)
    {
      if ( !_mesh )
        dolfin_error("Mesh has not been specified, unable to initialize mesh function.");
      init(*_mesh, dim, size);
    }

    /// Initialize mesh function for given topological dimension
    void init(Mesh& mesh, uint dim)
    {
      init(mesh, dim, mesh.size(dim));
    }

    /// Initialize mesh function for given topological dimension of given size
    void init(Mesh& mesh, uint dim, uint size)
    {
      // Initialize mesh for entities of given dimension
      mesh.init(dim);
      dolfin_assert(mesh.size(dim) == size);
      
      // Initialize data
      _mesh = &mesh;
      _dim = dim;
      _size = size;
      if ( _values )
	delete [] _values;
      _values = new T[size];
    }

    /// Get value at given entity
    inline T get(const MeshEntity& entity) const
    {
      dolfin_assert(_values);
      dolfin_assert(&entity.mesh() == _mesh);
      dolfin_assert(entity.dim() == _dim);
      dolfin_assert(entity.index() < _size);
      return _values[entity.index()];
    }

    /// Get value at given entity
    inline T get(uint index) const
    {
      dolfin_assert(_values);
      dolfin_assert(index < _size);
      return _values[index];
    }

    /// Set value at given entity
    inline void set(const MeshEntity& entity, const T& value)
    {
      dolfin_assert(_values);
      dolfin_assert(&entity.mesh() == _mesh);
      dolfin_assert(entity.dim() == _dim);
      dolfin_assert(entity.index() < _size);
      _values[entity.index()] = value;
    }
    
    /// Set value at given entity
    inline void set(uint index, const T& value)
    {
      dolfin_assert(_values);
      dolfin_assert(index < _size);
      _values[index] = value;
    }
    
    /// Display mesh function data
    void disp() const
    {
      cout << "Mesh function data" << endl;
      cout << "------------------" << endl;
      dolfin_begin();
      cout << "Topological dimension: " << _dim << endl;
      cout << "Number of values:      " << _size << endl;
      cout << endl;
      for (uint i = 0; i < _size; i++)
      {
        cout << "(" << _dim << ", " << i << "): " << _values[i] << endl;
      }
      dolfin_end();
    }

  private:

    /// Values at the set of mesh entities
    T* _values;

    /// The mesh
    Mesh* _mesh;

    /// Topological dimension
    uint _dim;

    /// Number of mesh entities
    uint _size;

  };

}

#endif
