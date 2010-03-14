// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman, 2007.
// Modified by Garth N. Wells, 2010.
//
// First added:  2006-05-22
// Last changed: 2010-01-06

#ifndef __MESH_FUNCTION_H
#define __MESH_FUNCTION_H

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/io/File.h>
#include <dolfin/mesh/Mesh.h>
#include "MeshEntity.h"

namespace dolfin
{

  class File;
  class XMLMeshFunction;

  /// A MeshFunction is a function that can be evaluated at a set of
  /// mesh entities. A MeshFunction is discrete and is only defined
  /// at the set of mesh entities of a fixed topological dimension.
  /// A MeshFunction may for example be used to store a global
  /// numbering scheme for the entities of a (parallel) mesh, marking
  /// sub domains or boolean markers for mesh refinement.

  template <class T> class MeshFunction : public Variable
  {
  public:

    /// Create empty mesh function
    MeshFunction() :
      Variable("f", "unnamed MeshFunction"),
      _values(0), _mesh(0), _dim(0), _size(0)
    {}

    /// Create empty mesh function on given mesh
    MeshFunction(const Mesh& mesh) :
      Variable("f", "unnamed MeshFunction"),
      _values(0), _mesh(&mesh), _dim(0), _size(0)
    {}

    /// Create mesh function on given mesh of given dimension
    MeshFunction(const Mesh& mesh, uint dim) :
      Variable("f", "unnamed MeshFunction"),
      _values(0), _mesh(&mesh), _dim(0), _size(0)
    {
      init(dim);
    }

    /// Create mesh function on given mesh of given dimension and initialise
    /// to a value
    MeshFunction(const Mesh& mesh, uint dim, const T& value) :
      Variable("f", "unnamed MeshFunction"),
      _values(0), _mesh(&mesh), _dim(0), _size(0)
    {
      init(dim);
      set_all(value);
    }

    /// Create function from data file
    MeshFunction(const Mesh& mesh, const std::string filename) :
      Variable("f", "unnamed MeshFunction"),
      _values(0), _mesh(&mesh), _dim(0), _size(0)
    {
      File file(filename);
      file >> *this;
    }

    /// Copy constructor
    MeshFunction(const MeshFunction<T>& f) :
      Variable("f", "unnamed MeshFunction"),
      _values(0), _mesh(0), _dim(0), _size(0)
    {
      *this = f;
    }

    /// Destructor
    ~MeshFunction()
    {
      delete [] _values;
    }

    /// Return mesh associated with mesh function
    const Mesh& mesh() const { assert(_mesh); return *_mesh; }

    /// Return topological dimension
    uint dim() const { return _dim; }

    /// Return size (number of entities)
    uint size() const { return _size; }

    /// Return array of values
    const T* values() const { return _values; }

    /// Return array of values
    T* values() { return _values; }

    /// Return value at given entity
    T& operator[] (const MeshEntity& entity)
    {
      assert(_values);
      assert(&entity.mesh() == _mesh);
      assert(entity.dim() == _dim);
      assert(entity.index() < _size);
      return _values[entity.index()];
    }

    /// Return value at given entity (const version)
    const T& operator[] (const MeshEntity& entity) const
    {
      assert(_values);
      assert(&entity.mesh() == _mesh);
      assert(entity.dim() == _dim);
      assert(entity.index() < _size);
      return _values[entity.index()];
    }

    /// Return value at given index
    T& operator[] (uint index)
    {
      assert(_values);
      assert(index < _size);
      return _values[index];
    }

    /// Return value at given index  (const version)
    const T& operator[] (uint index) const
    {
      assert(_values);
      assert(index < _size);
      return _values[index];
    }

    /// Assign mesh function
    const MeshFunction<T>& operator= (const MeshFunction<T>& f)
    {
      _mesh = f._mesh;
      _dim = f._dim;
      _size = f._size;
      delete [] _values;
      _values = new T[_size];
      for (uint i = 0; i < _size; i++)
        _values[i] = f._values[i];
      return *this;
    }

    /// Set all values to given value
    const MeshFunction<T>& operator= (const T& value)
    {
      set_all(value);
      return *this;
    }

    /// Initialize mesh function for given topological dimension
    void init(uint dim)
    {
      if (!_mesh)
        error("Mesh has not been specified, unable to initialize mesh function.");
      _mesh->init(dim);
      init(*_mesh, dim, _mesh->size(dim));
    }

    /// Initialize mesh function for given topological dimension of given size
    void init(uint dim, uint size)
    {
      if (!_mesh)
        error("Mesh has not been specified, unable to initialize mesh function.");
      _mesh->init(dim);
      init(*_mesh, dim, size);
    }

    /// Initialize mesh function for given topological dimension
    void init(const Mesh& mesh, uint dim)
    {
      mesh.init(dim);
      init(mesh, dim, mesh.size(dim));
    }

    /// Initialize mesh function for given topological dimension of given size
    void init(const Mesh& mesh, uint dim, uint size)
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

    /// Set all values to given value
    void set_all(const T& value)
    {
      assert(_values);
      for (uint i = 0; i < _size; i++)
        _values[i] = value;
    }

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const
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
        s << "<MeshFuncton of topological dimension " << _dim << " containing " << _size << " values>";

      return s.str();
    }

    // Input and output
    typedef XMLMeshFunction XMLHandler;

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

}

#endif
