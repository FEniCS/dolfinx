// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-22
// Last changed: 2006-05-22

#ifndef __MESH_FUNCTION_H
#define __MESH_FUNCTION_H

#include <dolfin/constants.h>

namespace dolfin
{

  /// A MeshFunction is a discrete function that can be evaluated at
  /// a set of mesh entities. Note that a MeshFunction is discrete and
  /// only defined at the set of mesh entities of a fixed topological
  /// dimension. A MeshFunction may for example be used to store a
  /// global numbering scheme for the entities for a (parallel) mesh,
  /// marking sub domains or boolean markers for mesh refinement.

  template <class T> class MeshFunction
  {
  public:

    /// Create empty mesh function on given mesh
    MeshFunction() : values(0), size(0) {}

    /// Destructor
    ~MeshFunction()
    {
      if ( values )
	delete [] values;
    }

    /// Evaluate mesh function at given entity
    inline T& operator() (uint e)
    {
      dolfin_assert(values && e < size);
      return values[e];
    }

    /// Evaluate mesh function at given entity
    inline const T& operator() (uint e) const
    {
      dolfin_assert(values && e < size);
      return values[e];
    }

    /// Initialize mesh function to given size
    void init(uint size)
    {
      this->size = size;
      if ( values )
	delete [] values;
      values = new T[size];
    }
    
  private:

    /// Values at the set of mesh entities
    T* values;

    /// Number of mesh entities
    uint size;

  };

}

#endif
