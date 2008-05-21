// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-19
// Last changed: 2008-05-19

#ifndef __MESH_DATA_H
#define __MESH_DATA_H

#include <map>

#include <dolfin/common/types.h>
#include "MeshFunction.h"

namespace dolfin
{

  class Mesh;

  /// The class MeshData is a container for auxiliary mesh data,
  /// represented as MeshFunctions over topological mesh entities.
  /// Each MeshFunction is identified by a unique user-specified
  /// string.
  ///
  /// Currently, only uint-valued MeshFunctions are supported.

  class MeshData
  {
  public:
    
    /// Constructor
    MeshData(Mesh& mesh);

    /// Destructor
    ~MeshData();

    /// Clear all data
    void clear();

    /// Create data with given name on entities of given dimension
    MeshFunction<uint>* create(std::string name, uint dim);

    /// Return data for given name
    MeshFunction<uint>* operator[] (std::string name);

    /// Display data
    void disp() const;

  private:

    // The mesh
    Mesh& mesh;

    // A map from named mesh data to MeshFunctions
    std::map<std::string, MeshFunction<uint>*> data;

  };

}

#endif
