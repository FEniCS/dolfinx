// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1
//
// Modified by Anders Logg, 2008.
// Modified by Garth N. Wells, 2010.
//
// First added:  2008-12-12
// Last changed: 2010-01-02
//
// This demo illustrates basic usage of block matrices and vectors.

#include <dolfin.h>
#include "StiffnessMatrix.h"

using namespace dolfin;

int main()
{
  // Create mesh
  UnitSquare mesh(4, 4);

  // Create a simple stiffness matrix
  Matrix A;
  StiffnessMatrix::FunctionSpace V(mesh);
  StiffnessMatrix::BilinearForm a(V, V);
  assemble(A, a);

  // Create a block matrix
  BlockMatrix AA(2, 2);
  AA(0, 0) = A;
  AA(1, 0) = A;
  AA(0, 1) = A;
  AA(1, 1) = A;

  // Create a block vector
  Vector x(A.size(0));
  for (unsigned int i = 0; i < x.size(); i++)
    x.setitem(i, i);
  BlockVector xx(2);
  xx(0) = x;
  xx(1) = x;

  // Create another block vector
  Vector y(A.size(1));
  BlockVector yy(2);
  yy(0) = y;
  yy(1) = y;

  // Multiply
  AA.mult(xx,yy);
  info("||Ax|| = %g", y.norm("l2"));
};
