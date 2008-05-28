// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-19
// Last changed: 2008-05-28

#ifndef __MESH_DATA_H
#define __MESH_DATA_H

#include <map>

#include <dolfin/common/types.h>
#include <dolfin/common/Array.h>
#include "MeshFunction.h"

namespace dolfin
{

  class Mesh;

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

    /// Clear all data
    void clear();

    /// Create MeshFunction with given name on entities of given dimension
    MeshFunction<uint>* createMeshFunction(std::string name, uint dim);

    /// Create Array with given name and size
    Array<uint>* createArray(std::string name, uint size);
    
    /// Return MeshFunction with given name (returning zero if data is not available)
    MeshFunction<uint>* meshFunction(std::string name);

    /// Return Array with given name (returning zero if data is not available)
    Array<uint>* array(std::string name);

    /// Display data
    void disp() const;

  private:

    // The mesh
    Mesh& mesh;

    // A map from named mesh data to MeshFunctions
    std::map<std::string, MeshFunction<uint>*> meshfunctions;

    // A map from named mesh data to Arrays
    std::map<std::string, Array<uint>*> arrays;

  };

}

#endif
