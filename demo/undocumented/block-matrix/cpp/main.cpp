// Copyright (C) 2008 Kent-Andre Mardal
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2008.
// Modified by Garth N. Wells, 2010-2011.
//
// First added:  2008-12-12
// Last changed: 2011-01-22
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
  StiffnessMatrix::FunctionSpace V(mesh);
  StiffnessMatrix::BilinearForm a(V, V);
  boost::shared_ptr<GenericMatrix> A(new Matrix);
  assemble(*A, a);

  // Create a block matrix
  BlockMatrix AA(2, 2);
  AA.set_block(0, 0, A);
  AA.set_block(1, 0, A);
  AA.set_block(0, 1, A);
  AA.set_block(1, 1, A);

  // Create vector
  boost::shared_ptr<GenericVector> x(new Vector(A->size(0)));
  Array<double> values(x->local_size());
  const unsigned int offset = x->local_range().first;
  for (unsigned int i = 0; i < x->local_size(); ++i)
    values[i] = i + offset;
  x->set_local(values);
  x->apply("add");

  // Create block vector
  BlockVector xx(2);
  xx.set_block(0, x);
  xx.set_block(1, x);

  // Create another block vector
  boost::shared_ptr<GenericVector> y(new Vector(A->size(1)));
  BlockVector yy(2);
  yy.set_block(0, y);
  yy.set_block(1, y);

  // Multiply
  AA.mult(xx, yy);
  info("||Ax|| = %g", y->norm("l2"));
}
