// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-08-21
// Last changed: 2006-08-21

#ifndef __MASS_MATRIX_H
#define __MASS_MATRIX_H

#include <dolfin/constants.h>
#include <dolfin/Matrix.h>
#include <dolfin/PETScMatrix.h>
#include <dolfin/uBlasSparseMatrix.h>
#include <dolfin/MatrixFactory.h>

namespace dolfin
{

  class Mesh;

  /// This class represents the standard mass matrix on a given mesh,
  /// represented as a default DOLFIN matrix.

  class MassMatrix : public Matrix
  {
  public:
  
    /// Construct mass matrix on a given mesh
    MassMatrix(Mesh& mesh) : Matrix()
    {
      MatrixFactory::computeMassMatrix(*this, mesh);
    }

  };

#ifdef HAVE_PETSC_H

  /// This class represents the standard mass matrix on a given mesh,
  /// represented as a DOLFIN PETSc matrix.

  class PETScMassMatrix : public PETScMatrix
  {
  public:
  
    /// Construct mass matrix on a given mesh
    PETScMassMatrix(Mesh& mesh) : PETScMatrix()
    {
      MatrixFactory::computeMassMatrix(*this, mesh);
    }

  };

#endif

  /// This class represents the standard mass matrix on a given mesh,
  /// represented as a sparse DOLFIN uBlas matrix.

  class uBlasMassMatrix : public uBlasSparseMatrix
  {
  public:
  
    /// Construct mass matrix on a given mesh
    uBlasMassMatrix(Mesh& mesh) : uBlasSparseMatrix()
    {
      MatrixFactory::computeMassMatrix(*this, mesh);
    }

  };

}

#endif
