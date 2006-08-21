// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-08-21
// Last changed: 2006-08-21

#ifndef __STIFFNESS_MATRIX_H
#define __STIFFNESS_MATRIX_H

#include <dolfin/constants.h>
#include <dolfin/Matrix.h>
#include <dolfin/PETScMatrix.h>
#include <dolfin/uBlasSparseMatrix.h>
#include <dolfin/MatrixFactory.h>

namespace dolfin
{

  class Mesh;

  /// This class represents the standard stiffness matrix for
  /// homogeneous Neumann boundary conditions on a given mesh,
  /// represented in the default DOLFIN matrix format.

  class StiffnessMatrix : public Matrix
  {
  public:
  
    /// Construct stiffness matrix with diffusivity c on a given mesh
    StiffnessMatrix(Mesh& mesh, real c = 1.0) : Matrix()
    {
      MatrixFactory::computeStiffnessMatrix(*this, mesh, c);
    }

  };

#ifdef HAVE_PETSC_H

  /// This class represents the standard stiffness matrix for
  /// homogeneous Neumann boundary conditions on a given mesh,
  /// represented as a DOLFIN PETSc matrix.

  class PETScStiffnessMatrix : public PETScMatrix
  {
  public:
  
    /// Construct stiffness matrix with diffusivity c on a given mesh
    PETScStiffnessMatrix(Mesh& mesh, real c = 1.0) : PETScMatrix()
    {
      MatrixFactory::computeStiffnessMatrix(*this, mesh, c);
    }

  };

#endif

  /// This class represents the standard stiffness matrix for
  /// homogeneous Neumann boundary conditions on a given mesh,
  /// represented as a sparse DOLFIN uBlas matrix.

  class uBlasStiffnessMatrix : public uBlasSparseMatrix
  {
  public:
  
    /// Construct stiffness matrix with diffusivity c on a given mesh
    uBlasStiffnessMatrix(Mesh& mesh, real c = 1.0) : uBlasSparseMatrix()
    {
      MatrixFactory::computeStiffnessMatrix(*this, mesh, c);
    }

  };

}

#endif
