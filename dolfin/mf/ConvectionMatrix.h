// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-08-21
// Last changed: 2006-08-21

#ifndef __CONVECTION_MATRIX_H
#define __CONVECTION_MATRIX_H

#include <dolfin/common/types.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/uBLASSparseMatrix.h>
#include "MatrixFactory.h"

namespace dolfin
{

  class Mesh;

  /// This class represents the standard convection matrix with
  /// constant velocity field c = (cx, cy) or c = (cx, cy, cz) on a
  /// given mesh, represented in the default DOLFIN matrix format.

  class ConvectionMatrix : public Matrix
  {
  public:
  
    /// Construct convection matrix with constant velocity c on a given mesh
    ConvectionMatrix(Mesh& mesh, double cx = 1.0, double cy = 0.0, double cz = 0.0) : Matrix()
    {
      MatrixFactory::computeConvectionMatrix(*this, mesh, cx, cy, cz);
    }

  };

#ifdef HAS_PETSC

  /// This class represents the standard convection matrix with
  /// constant velocity field c = (cx, cy) or c = (cx, cy, cz) on a
  /// given mesh, represented as a DOLFIN PETSc matrix.

  class PETScConvectionMatrix : public PETScMatrix
  {
  public:
  
    /// Construct convection matrix with constant velocity c on a given mesh
    PETScConvectionMatrix(Mesh& mesh, double cx = 1.0, double cy = 0.0, double cz = 0.0) : PETScMatrix()
    {
      MatrixFactory::computeConvectionMatrix(*this, mesh, cx, cy, cz);
    }

  };

#endif

  /// This class represents the standard convection matrix with
  /// constant velocity field c = (cx, cy) or c = (cx, cy, cz) on a
  /// given mesh, represented as a sparse DOLFIN uBLAS matrix.

  class uBLASConvectionMatrix : public uBLASSparseMatrix
  {
  public:
  
    /// Construct convection matrix with constant velocity c on a given mesh
    uBLASConvectionMatrix(Mesh& mesh, double cx = 1.0, double cy = 0.0, double cz = 0.0) : uBLASSparseMatrix()
    {
      MatrixFactory::computeConvectionMatrix(*this, mesh, cx, cy, cz);
    }

  };

}

#endif
