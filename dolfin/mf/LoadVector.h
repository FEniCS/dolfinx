// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-08-21
// Last changed: 2006-08-21

#ifndef __LOAD_VECTOR_H
#define __LOAD_VECTOR_H

#include <dolfin/common/types.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/uBlasVector.h>
#include "MatrixFactory.h"

namespace dolfin
{

  class Mesh;

  /// This class represents the standard load vector with constant
  /// load c on a given mesh, represented as a default DOLFIN vector.

  class LoadVector : public Vector
  {
  public:
  
    /// Construct load vector with constant load c on a given mesh
    LoadVector(Mesh& mesh, real c = 1.0) : Vector()
    {
      MatrixFactory::computeLoadVector(*this, mesh, c);
    }

  };

#ifdef HAS_PETSC

  /// This class represents the standard load vector with constant
  /// load c on a given mesh, represented as a DOLFIN PETSc vector.

  class PETScLoadVector : public PETScVector
  {
  public:
  
    /// Construct load vector with constant load c on a given mesh
    PETScLoadVector(Mesh& mesh, real c = 1.0) : PETScVector()
    {
      MatrixFactory::computeLoadVector(*this, mesh, c);
    }

  };

#endif

  /// This class represents the standard load vector with constant
  /// load c on a given mesh, represented as a sparse DOLFIN uBlas
  /// vector.

  class uBlasLoadVector : public uBlasVector
  {
  public:
  
    /// Construct load vector with constant load c on a given mesh
    uBlasLoadVector(Mesh& mesh, real c = 1.0) : uBlasVector()
    {
      MatrixFactory::computeLoadVector(*this, mesh, c);
    }

  };

}

#endif
