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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2008-2012
// Modified by Garth N. Wells 2011
//
// First added:  2008-08-25
// Last changed: 2012-03-15

#include <iostream>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Timer.h>
#include <dolfin/common/NoDeleter.h>
#include "dolfin/common/utils.h"
#include "BlockVector.h"
#include "DefaultFactory.h"
#include "GenericVector.h"
#include "Matrix.h"
#include "BlockMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BlockMatrix::BlockMatrix(uint m, uint n) : matrices(boost::extents[m][n])
{
  for (uint i = 0; i < m; i++)
    for (uint j = 0; j < n; j++)
      matrices[i][j].reset(new Matrix());
}
//-----------------------------------------------------------------------------
BlockMatrix::~BlockMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BlockMatrix::set_block(uint i, uint j, boost::shared_ptr<GenericMatrix> m)
{
  dolfin_assert(i < matrices.shape()[0]);
  dolfin_assert(j < matrices.shape()[1]);
  matrices[i][j] = m;
}
//-----------------------------------------------------------------------------
const boost::shared_ptr<GenericMatrix> BlockMatrix::get_block(uint i, uint j) const
{
  dolfin_assert(i < matrices.shape()[0]);
  dolfin_assert(j < matrices.shape()[1]);
  return matrices[i][j];
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericMatrix> BlockMatrix::get_block(uint i, uint j)
{
  dolfin_assert(i < matrices.shape()[0]);
  dolfin_assert(j < matrices.shape()[1]);
  return matrices[i][j];
}
//-----------------------------------------------------------------------------
std::size_t BlockMatrix::size(uint dim) const
{
  dolfin_assert(dim < 1);
  return matrices.shape()[dim];
}
//-----------------------------------------------------------------------------
void BlockMatrix::zero()
{
  for(uint i = 0; i < matrices.shape()[0]; i++)
    for(uint j = 0; j < matrices.shape()[1]; j++)
      matrices[i][j]->zero();
}
//-----------------------------------------------------------------------------
void BlockMatrix::apply(std::string mode)
{
  Timer("Apply (matrix)");
  for (uint i = 0; i < matrices.shape()[0]; i++)
    for (uint j = 0; j < matrices.shape()[1]; j++)
      matrices[i][j]->apply(mode);
}
//-----------------------------------------------------------------------------
std::string BlockMatrix::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    for (uint i = 0; i < matrices.shape()[0]; i++)
    {
      for (uint j = 0; i < matrices.shape()[1]; j++)
      {
        s << "  BlockMatrix (" << i << ", " << j << ")" << std::endl << std::endl;
        s << indent(indent(matrices[i][j]->str(true))) << std::endl;
      }
    }
  }
  else
    s << "<BlockMatrix containing " << matrices.shape()[0] << " x " << matrices.shape()[1] << " blocks>";

  return s.str();
}
//-----------------------------------------------------------------------------
void BlockMatrix::mult(const BlockVector& x, BlockVector& y,
                       bool transposed) const
{
  if (transposed)
  {
    dolfin_error("BlockMatrix.cpp",
                 "compute transpose matrix-vector product",
                 "Not implemented for block matrices");
  }

  // Create tempory vector
  dolfin_assert(matrices[0][0]);
  boost::shared_ptr<GenericVector>
    z_tmp = matrices[0][0]->factory().create_vector();

  // Loop over block rows
  for(std::size_t row = 0; row < matrices.shape()[0]; row++)
  {
    // RHS sub-vector
    GenericVector& _y = *(y.get_block(row));

    // Resize y and zero
    dolfin_assert(matrices[row][0]);
    const GenericMatrix& _A = *matrices[row][0];
    _A.resize(_y, 0);
    _y.zero();

    // Resize z_tmp and zero
    z_tmp->resize(_y.size());
    _A.resize(*z_tmp, 0);

    // Loop over block columns
    for(std::size_t col = 0; col < matrices.shape()[1]; ++col)
    {
      const GenericVector& _x = *(x.get_block(col));
      dolfin_assert(matrices[row][col]);
      matrices[row][col]->mult(_x, *z_tmp);
      _y += *z_tmp;
    }
  }
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericMatrix> BlockMatrix::schur_approximation(bool symmetry) const
{
  // Currently returns [diag(C * diag(A)^-1 * B) - D]
  if (!symmetry)
  {
    dolfin_error("BlockMatrix.cpp",
                 "compute Schur complement approximation",
                 "Not implemented for unsymmetric matrix");
  }

  dolfin_assert(matrices.size()==2 && matrices[0].size()==2 && matrices[1].size()==2);

  GenericMatrix &A = *matrices[0][0];
  GenericMatrix &C = *matrices[1][0];
  GenericMatrix &D = *matrices[1][1];

  boost::shared_ptr<GenericMatrix> S(D.copy());

  std::vector<std::size_t> cols_i;
  std::vector<double> vals_i;
  for (std::size_t i = 0; i < D.size(0); i++)
  {
    C.getrow(i, cols_i, vals_i);
    double diag_ii = 0;
    for (std::size_t k = 0; k < cols_i.size(); k++)
    {
      const std::size_t j = cols_i[k];
      const double val=vals_i[k];
      diag_ii -= val*val/A(j,j);
    }
    const DolfinIndex _i = i;
    S->add(&diag_ii, 1, &_i, 1, &_i);
  }
  return S;
}
