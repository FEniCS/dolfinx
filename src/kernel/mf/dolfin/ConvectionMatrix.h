// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-08-21
// Last changed: 2006-08-21

#ifndef __CONVECTION_MATRIX_H
#define __CONVECTION_MATRIX_H

#include <dolfin/constants.h>
#include <dolfin/Matrix.h>
#include <dolfin/PETScMatrix.h>
#include <dolfin/uBlasSparseMatrix.h>
#include <dolfin/MatrixFactory.h>

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
    ConvectionMatrix(Mesh& mesh, real cx = 1.0, real cy = 0.0, real cz = 0.0) : Matrix()
    {
      MatrixFactory::computeConvectionMatrix(*this, mesh, cx, cy, cz);
    }

  };

#ifdef HAVE_PETSC_H

  /// This class represents the standard convection matrix with
  /// constant velocity field c = (cx, cy) or c = (cx, cy, cz) on a
  /// given mesh, represented as a DOLFIN PETSc matrix.

  class PETScConvectionMatrix : public PETScMatrix
  {
  public:
  
    /// Construct convection matrix with constant velocity c on a given mesh
    PETScConvectionMatrix(Mesh& mesh, real cx = 1.0, real cy = 0.0, real cz = 0.0) : PETScMatrix()
    {
      MatrixFactory::computeConvectionMatrix(*this, mesh, cx, cy, cz);
    }

  };

#endif

  /// This class represents the standard convection matrix with
  /// constant velocity field c = (cx, cy) or c = (cx, cy, cz) on a
  /// given mesh, represented as a sparse DOLFIN uBlas matrix.

  class uBlasConvectionMatrix : public uBlasSparseMatrix
  {
  public:
  
    /// Construct convection matrix with constant velocity c on a given mesh
    uBlasConvectionMatrix(Mesh& mesh, real cx = 1.0, real cy = 0.0, real cz = 0.0) : uBlasSparseMatrix()
    {
      MatrixFactory::computeConvectionMatrix(*this, mesh, cx, cy, cz);
    }

  };

}

#endif
