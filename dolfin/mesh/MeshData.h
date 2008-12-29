// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson, 2008.
//
// First added:  2008-05-19
// Last changed: 2008-12-29

#ifndef __MESH_DATA_H
#define __MESH_DATA_H

#include <map>
#include <dolfin/common/types.h>

namespace dolfin
{

  class Mesh;
  template <class T> class MeshFunction;
  template <class T> class Array;

  /// The class MeshData is a container for auxiliary mesh data,
  /// represented either as MeshFunctions over topological mesh
  /// entities or Arrays. Each dataset is identified by a unique
  /// user-specified string.
  ///
  /// Currently, only uint-valued data is supported.

  class MeshData
  {
  public:
    
    /// Constructor
    MeshData(Mesh& mesh);

    /// Destructor
    ~MeshData();

    /// Assignment operator
    const MeshData& operator= (const MeshData& data);

    /// Clear all data
    void clear();

    /// Create MeshFunction with given name (uninitialized)
    MeshFunction<uint>* createMeshFunction(std::string name);

    /// Create MeshFunction with given name and dimension
    MeshFunction<uint>* createMeshFunction(std::string name, uint dim);

    /// Create Array with given name and size
    Array<uint>* createArray(std::string name, uint size);

    /// Create map with given name and size
    std::map<uint, uint>* createMapping(std::string name);

    /// Return Array with given name (returning zero if data is not available)
    Array<uint>* array(const std::string name) const;
    
    /// Return MeshFunction with given name (returning zero if data is not available)
    MeshFunction<uint>* meshFunction(const std::string name) const;

    /// Return Map with given name (returning zero if data is not available)
    std::map<uint, uint>* mapping(const std::string name) const;

    /// Display data
    void disp() const;

    /// Friends
    friend class XMLFile;

  private:

    // The mesh
    Mesh& mesh;

    // A map from named mesh data to MeshFunctions
    std::map<std::string, MeshFunction<uint>*> meshfunctions;

    // A map from named mesh data to Arrays
    std::map<std::string, Array<uint>*> arrays;

    // A map from named mesh data to map
    std::map<std::string, std::map<uint, uint>*> maps;
    
  };

}

#endif
