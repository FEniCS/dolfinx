// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-22
// Last changed: 2006-11-22

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
    MeshFunction() :  _values(0), _mesh(0), _dim(0), _size(0) {}

    /// Destructor
    ~MeshFunction()
    {
      if ( _values )
	delete [] _values;
    }

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

    /// Initialize mesh function for given topological dimension
    void init(const Mesh& mesh, uint dim)
    {
      init(mesh, dim, mesh.size(dim));
    }

    /// Initialize mesh function for given topological dimension of given size
    void init(const Mesh& mesh, uint dim, uint size)
    {
      _mesh = &mesh;
      _dim = dim;
      _size = size;
      if ( _values )
	delete [] _values;
      _values = new T[size];
    }

    /// Set value at given entity
    inline void set(uint index, const T& value)
    {
      dolfin_assert(_values);
      dolfin_assert(index < _size);
      _values[index] = value;
    }
    
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

}

#endif
