// Copyright (C) 2008 Dag Lindbo
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either versio1n 3 of the License, or
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
// Modified by Garth N. Wells 2008-2010
// Modified by Anders Logg 2011-2012
//
// First added:  2008-07-06
// Last changed: 2012-03-15

#ifdef HAS_MTL4

#include "MTL4Vector.h"
#include <dolfin/common/Timer.h>
#include <dolfin/log/dolfin_log.h>
#include "TensorLayout.h"
#include "MTL4Matrix.h"
#include "MTL4Factory.h"

using namespace dolfin;
using namespace mtl;

//-----------------------------------------------------------------------------
MTL4Matrix::MTL4Matrix() : ins(0), nnz_row(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MTL4Matrix::MTL4Matrix(uint M, uint N, uint nz) : ins(0), nnz_row(nz)
{
  resize(M, N);
}
//-----------------------------------------------------------------------------
MTL4Matrix::MTL4Matrix(const MTL4Matrix& mat) : ins(0), nnz_row(0)
{
  dolfin_assert_no_inserter();

  // Deep copy
  A = mat.mat();
  nnz_row = mat.nnz_row;
}
//-----------------------------------------------------------------------------
MTL4Matrix::~MTL4Matrix()
{
  delete ins;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericMatrix> MTL4Matrix::copy() const
{
  dolfin_assert_no_inserter();
  boost::shared_ptr<GenericMatrix> A(new MTL4Matrix(*this));
  return A;
}
//-----------------------------------------------------------------------------
void MTL4Matrix::resize(uint M, uint N)
{
  dolfin_assert_no_inserter();
  A.change_dim(M, N);
}
//-----------------------------------------------------------------------------
void MTL4Matrix::init(const TensorLayout& tensor_layout)
{
  resize(tensor_layout.size(0), tensor_layout.size(1));
}
//-----------------------------------------------------------------------------
dolfin::uint MTL4Matrix::size(uint dim) const
{
  if (dim > 1)
  {
    dolfin_error("MTL4Matrix.cpp",
                 "access size of MTL4 matrix",
                 "Illegal axis (%d), must be 0 or 1", dim);
  }

  if (dim == 0)
    return mtl::matrix::num_rows(A);
  else if (dim == 1)
    return mtl::matrix::num_cols(A);

  return 0;
}
//-----------------------------------------------------------------------------
void MTL4Matrix::resize(GenericVector& z, uint dim) const
{
  z.resize(size(dim));
}
//-----------------------------------------------------------------------------
void MTL4Matrix::get(double* block, uint m, const uint* rows, uint n,
                     const uint* cols) const
{
  dolfin_assert_no_inserter();
  for (uint i = 0; i < m; i++)
    for (uint j = 0; j < n; j++)
      block[i*n+j] = A[rows[i]][cols[j]];
}
//-----------------------------------------------------------------------------
void MTL4Matrix::set(const double* block, uint m, const uint* rows, uint n,
                     const uint* cols)
{
  if (!ins)
    init_inserter(nnz_row);

  for (uint i = 0; i < m; i++)
    for (uint j = 0; j < n; j++)
      (*ins)[rows[i]][cols[j]] = block[i*n +j];
}
//-----------------------------------------------------------------------------
void MTL4Matrix::add(const double* block, uint m, const uint* rows, uint n,
                     const uint* cols)
{
  // This is not thread-safe
  if (!ins)
    init_inserter(nnz_row);

  // Block insertion
  //*ins << element_array(mtl::dense2D<double>(m, n, const_cast<double*>(block)),
  //                      mtl::dense_vector<uint>(m, const_cast<uint*>(rows)),
  //                      mtl::dense_vector<uint>(n, const_cast<uint*>(cols)));

  for (uint i = 0; i < m; i++)
    for (uint j = 0; j < n; j++)
      (*ins)[rows[i]][cols[j]] << block[i*n +j];
}
//-----------------------------------------------------------------------------
void MTL4Matrix::axpy(double a, const GenericMatrix& A,
                      bool same_nonzero_pattern)
{
  // Check for same size
  if (size(0) != A.size(0) or size(1) != A.size(1))
  {
    dolfin_error("MTL4Matrix.cpp",
                 "perform axpy operation with MTL4 matrix",
                 "Non-matching matrix dimensions for axpy operation");
  }

  // Do we need to check for same sparsity pattern?
  this->A += (a)*(A.down_cast<MTL4Matrix>().mat());
}
//-----------------------------------------------------------------------------
double MTL4Matrix::norm(std::string norm_type) const
{
  if (norm_type == "l1")
    return one_norm(A);
  else if (norm_type == "linf")
    return infinity_norm(A);
  else if (norm_type == "frobenius")
    return frobenius_norm(A);
  else
  {
    dolfin_error("MTL4Matrix.cpp",
                 "compute norm of MTL4 matrix",
                 "Unknown norm type (\"%s\")", norm_type.c_str());
    return 0.0;
  }
}
//-----------------------------------------------------------------------------
void MTL4Matrix::zero()
{
  dolfin_assert_no_inserter();
  A *= 0;
}
//-----------------------------------------------------------------------------
void MTL4Matrix::apply(std::string mode)
{
  Timer("Apply (matrix)");
  if (ins)
    delete ins;
  ins = 0;
}
//-----------------------------------------------------------------------------
std::string MTL4Matrix::str(bool verbose) const
{
  dolfin_assert_no_inserter();

  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    s << A;
  }
  else
    s << "<MTL4Matrix of size " << size(0) << " x " << size(1) << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void MTL4Matrix::ident(uint m, const uint* rows)
{
  // Zero rows
  zero(m, rows);

  // Place one on the diagonal
  mtl::matrix::inserter< mtl4_sparse_matrix > ins_A(A);
  for(uint i = 0; i < m ; ++i)
    ins_A[ rows[i] ][ rows[i] ] << 1.0;
}
//-----------------------------------------------------------------------------
void MTL4Matrix::zero(uint m, const uint* rows)
{
  dolfin_assert_no_inserter();

  // Define cursors
  typedef mtl::traits::range_generator<mtl::tag::row, mtl4_sparse_matrix >::type c_type;
  typedef mtl::traits::range_generator<mtl::tag::nz, c_type>::type  ic_type;

  mtl::traits::value<mtl4_sparse_matrix >::type value(A);

  // Create row cursors
  c_type cursor(mtl::begin<mtl::tag::row>(A));
  c_type cend(mtl::end<mtl::tag::row>(A));

  for(uint i = 0; i < m; ++i)
  {
    // Increment cursor
    if (i == 0)
      cursor += rows[i];
    else
      cursor += rows[i] - rows[i-1];

    // Check that we haven't gone beyond last row
    dolfin_assert(*cursor <= *cend);

    // Zero row
    for (ic_type icursor(mtl::begin<mtl::tag::nz>(cursor)),
                         icend(mtl::end<mtl::tag::nz>(cursor));
                         icursor != icend; ++icursor)
    {
      value(*icursor, 0.0);
    }
  }
}
//-----------------------------------------------------------------------------
void MTL4Matrix::mult(const GenericVector& x_, GenericVector& Ax_) const
{
  dolfin_assert_no_inserter();
  Ax_.down_cast<MTL4Vector>().vec() = A*x_.down_cast<MTL4Vector>().vec();
}
//-----------------------------------------------------------------------------
void MTL4Matrix::transpmult(const GenericVector& x_, GenericVector& Ax_) const
{
  dolfin_assert_no_inserter();
  Ax_.down_cast<MTL4Vector>().vec() = trans(this->A)*x_.down_cast<MTL4Vector>().vec();
}
//-----------------------------------------------------------------------------
void MTL4Matrix::getrow(uint row_idx, std::vector<uint>& columns,
                        std::vector<double>& values) const
{
  dolfin_assert_no_inserter();
  dolfin_assert(row_idx < this->size(0));

  columns.clear();
  values.clear();

  // Define cursors
  typedef mtl::traits::range_generator<mtl::tag::row, mtl4_sparse_matrix >::type c_type;
  typedef mtl::traits::range_generator<mtl::tag::nz, c_type>::type  ic_type;

  mtl::traits::col<mtl4_sparse_matrix >::type col(A);
  mtl::traits::const_value<mtl4_sparse_matrix >::type value(A);

  // Row cursors
  c_type cursor(mtl::begin<mtl::tag::row>(A));
  cursor += row_idx;

  for (ic_type icursor(mtl::begin<mtl::tag::nz>(cursor)), icend(mtl::end<mtl::tag::nz>(cursor)); icursor != icend; ++icursor)
  {
    columns.push_back(col(*icursor));
    values.push_back(value(*icursor));
  }
}
//-----------------------------------------------------------------------------
void MTL4Matrix::setrow(uint row, const std::vector<uint>& columns, const std::vector<double>& values)
{
  dolfin_assert(columns.size() == values.size());
  dolfin_assert(row < this->size(0));

  if (!ins)
    init_inserter(nnz_row);

  for (uint i=0; i<columns.size(); i++)
    (*ins)[row][columns[i] ] = values[i];
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& MTL4Matrix::factory() const
{
  return MTL4Factory::instance();
}
//-----------------------------------------------------------------------------
const mtl4_sparse_matrix& MTL4Matrix::mat() const
{
  dolfin_assert_no_inserter();
  return A;
}
//-----------------------------------------------------------------------------
mtl4_sparse_matrix& MTL4Matrix::mat()
{
  dolfin_assert_no_inserter();
  return A;
}
//-----------------------------------------------------------------------------
const MTL4Matrix& MTL4Matrix::operator*= (double a)
{
  dolfin_assert_no_inserter();
  A *= a;
  return *this;
}
//-----------------------------------------------------------------------------
const MTL4Matrix& MTL4Matrix::operator/= (double a)
{
  dolfin_assert_no_inserter();
  A /= a;
  return *this;
}
//-----------------------------------------------------------------------------
const GenericMatrix& MTL4Matrix::operator= (const GenericMatrix& A)
{
  *this = A.down_cast<MTL4Matrix>();
  return *this;
}
//-----------------------------------------------------------------------------
const MTL4Matrix& MTL4Matrix::operator= (const MTL4Matrix& A)
{
  // Check for self-assignment
  if (this != &A)
    this->A = A.mat();

  nnz_row = A.nnz_row;
  return *this;
}
//-----------------------------------------------------------------------------
boost::tuples::tuple<const std::size_t*, const std::size_t*, const double*, int> MTL4Matrix::data() const
{
  dolfin_assert_no_inserter();
  typedef boost::tuples::tuple<const std::size_t*, const std::size_t*, const double*, int> tuple;
  return tuple(A.address_major(), A.address_minor(), A.address_data(), A.nnz());
}
//-----------------------------------------------------------------------------
void MTL4Matrix::init_inserter(uint nnz)
{
  dolfin_assert_no_inserter();
  if (nnz > 0)
    ins = new mtl::matrix::inserter<mtl4_sparse_matrix, mtl::update_plus<double> >(A, nnz);
  else
    ins = new mtl::matrix::inserter<mtl4_sparse_matrix, mtl::update_plus<double> >(A, 25);
}
//-----------------------------------------------------------------------------
inline void MTL4Matrix::dolfin_assert_no_inserter() const
{
  if (ins)
  {
    dolfin_error("MTL4Matrix.cpp",
                 "perform operation on MTL4 matrix",
                 "Operation not allowed while inserter active. You may have forgotten to call apply()");
  }
}
//-----------------------------------------------------------------------------

#endif
