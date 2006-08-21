// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-08-21
// Last changed: 2006-08-21

#ifndef __MATRIX_FACTORY_H
#define __MATRIX_FACTORY_H

namespace dolfin
{

  class GenericMatrix;
  class Mesh;

  /// This class provides functionality for computing a set of
  /// standard finite element matrices, such as the mass matrix
  /// and the stiffness matrix.

  class MatrixFactory
  {
  public:
    
    /// Compute mass matrix on a given mesh
    void computeMassMatrix(GenericMatrix& A, Mesh& mesh);

    /// Compute stiffness matrix with diffusivity c on a given mesh
    void computeStiffnessMatrix(GenericMatrix& A, Mesh& mesh, real c);
    
  };

}

#endif
