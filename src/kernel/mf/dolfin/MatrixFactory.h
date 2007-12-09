// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-08-21
// Last changed: 2006-08-21

#ifndef __MATRIX_FACTORY_H
#define __MATRIX_FACTORY_H

#include <dolfin/constants.h>

namespace dolfin
{

  class DofMapSet;
  class GenericMatrix;
  class GenericVector;
  class Mesh;

  /// This class provides functionality for computing a set of
  /// standard finite element matrices, such as the mass matrix
  /// and the stiffness matrix, with piecewise linear elements.
  /// For other matrices (forms) and elements, forms must be
  /// defined in the FFC form language and assembled.

  class MatrixFactory
  {
  public:
    
    /// Compute mass matrix on a given mesh
    static void computeMassMatrix(GenericMatrix& A, Mesh& mesh);

    /// Compute stiffness matrix with diffusivity c on a given mesh
    static void computeStiffnessMatrix(GenericMatrix& A, Mesh& mesh, real c = 1.0);
    
    /// Compute convection matrix with constant velocity c on a given mesh
    static void computeConvectionMatrix(GenericMatrix& A, Mesh& mesh,
					real cx = 1.0, real cy = 0.0, real cz = 0.0);
    
    /// Construct load vector with constant load c on a given mesh
    static void computeLoadVector(GenericVector& x, Mesh& mesh, real c = 1.0);

  };

}

#endif
