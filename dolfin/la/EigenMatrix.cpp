// Copyright (C) 2015 Chris Richardson
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

#include "EigenFactory.h"
#include "SparsityPattern.h"
#include "EigenMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& EigenMatrix::factory() const
{
  return EigenFactory::instance();
}
//---------------------------------------------------------------------------
EigenMatrix::EigenMatrix() : _matA(0, 0)
{
  // Do nothing
}
//---------------------------------------------------------------------------
EigenMatrix::EigenMatrix(std::size_t M, std::size_t N) : _matA(M, N)
{
  // Do nothing
}
//---------------------------------------------------------------------------
EigenMatrix::EigenMatrix(const EigenMatrix& A) : _matA(A._matA)
{
  // Do nothing
}
//---------------------------------------------------------------------------
EigenMatrix::~EigenMatrix()
{
  // Do nothing
}
//---------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> EigenMatrix::copy() const
{
  return std::shared_ptr<GenericMatrix>(new EigenMatrix(*this));
}
//---------------------------------------------------------------------------
void EigenMatrix::resize(std::size_t M, std::size_t N)
{
  // FIXME: Do we want to allow this?
  // Resize matrix
  if(size(0) != M || size(1) != N)
    _matA.resize(M, N);
}
//---------------------------------------------------------------------------
std::size_t EigenMatrix::size(std::size_t dim) const
{
  if (dim > 1)
  {
    dolfin_error("EigenMatrix.cpp",
                 "access size of Eigen matrix",
                 "Illegal axis (%d), must be 0 or 1", dim);
  }

  return (dim == 0 ? _matA.rows() : _matA.cols());
}
//---------------------------------------------------------------------------
double EigenMatrix::norm(std::string norm_type) const
{
  if (norm_type == "l1")
  {
    double _norm = 0.0;
    for (std::size_t i = 0; i < size(1); ++i)
      _norm = std::max(_norm, _matA.col(i).cwiseAbs().sum());
    return _norm;
  }
  else if (norm_type == "l2")
    return _matA.squaredNorm();
  else if (norm_type == "frobenius")
    return _matA.norm();
  else if (norm_type == "linf")
  {
    double _norm = 0.0;
    for (std::size_t i = 0; i < size(0); ++i)
      _norm = std::max(_norm, _matA.row(i).cwiseAbs().sum());
    return _norm;
  }
  else
  {
    dolfin_error("EigenMatrix.cpp",
                 "compute norm of Eigen matrix",
                 "Unknown norm type (\"%s\")",
                 norm_type.c_str());
    return 0.0;
  }
}
//---------------------------------------------------------------------------
void EigenMatrix::getrow(std::size_t row_idx,
                         std::vector<std::size_t>& columns,
                         std::vector<double>& values) const
{
  dolfin_assert(row_idx < this->size(0));

  // Check storage is RowMajor
  if (!eigen_matrix_type::IsRowMajor)
    dolfin_error("EigenMatrix.cpp",
                 "get row of Eigen matrix",
                 "Cannot get row from column major matrix");

  // Insert values into std::vectors
  columns.clear();
  values.clear();
  for (eigen_matrix_type::InnerIterator it(_matA, row_idx); it; ++it)
  {
    columns.push_back(it.index());
    values.push_back(it.value());
  }
}
//----------------------------------------------------------------------------
void EigenMatrix::setrow(std::size_t row_idx,
                         const std::vector<std::size_t>& columns,
                         const std::vector<double>& values)
{
  dolfin_assert(columns.size() == values.size());
  dolfin_assert(row_idx < this->size(0));
  for(std::size_t i = 0; i < columns.size(); i++)
    _matA.coeffRef(row_idx, columns[i]) = values[i];
}
//----------------------------------------------------------------------------
void EigenMatrix::init_vector(GenericVector& z, std::size_t dim) const
{
  z.init(mpi_comm(), size(dim));
}
//----------------------------------------------------------------------------
void EigenMatrix::set(const double* block, std::size_t m,
                      const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols)
{
  for (std::size_t i = 0; i < m; ++i)
  {
    const dolfin::la_index row = rows[i];
    for (std::size_t j = 0; j < n; ++j)
       _matA.coeffRef(row , cols[j]) = block[i*n + j];
  }
}
//---------------------------------------------------------------------------
void EigenMatrix::add(const double* block, std::size_t m,
                      const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols)
{
  for (std::size_t i = 0; i < m; ++i)
  {
    const dolfin::la_index row = rows[i];
    for (std::size_t j = 0; j < n; ++j)
       _matA.coeffRef(row , cols[j]) += block[i*n + j];
  }
}
//---------------------------------------------------------------------------
void EigenMatrix::get(double* block, std::size_t m,
                      const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols) const
{
  for (std::size_t i = 0; i < m; ++i)
  {
    const dolfin::la_index row = rows[i];
    for(std::size_t j = 0; j < n; ++j)
      block[i*n + j] = _matA.coeff(row, cols[j]);
  }
}
//---------------------------------------------------------------------------
void EigenMatrix::zero()
{
  // Set to zero whilst keeping the non-zero pattern
  for (dolfin::la_index i = 0; i < _matA.outerSize(); ++i)
    for (eigen_matrix_type::InnerIterator it(_matA, i); it; ++it)
      it.valueRef() = 0.0;
}
//----------------------------------------------------------------------------
void EigenMatrix::zero(std::size_t m, const dolfin::la_index* rows)
{
  for (const dolfin::la_index* i_ptr = rows; i_ptr != rows + m; ++i_ptr)
    for (eigen_matrix_type::InnerIterator it(_matA, *i_ptr); it; ++it)
      it.valueRef() = 0.0;
}
//----------------------------------------------------------------------------
void EigenMatrix::ident(std::size_t m, const dolfin::la_index* rows)
{
  bool diagonal_unset;
  const dolfin::la_index num_cols = size(1);

  // Loop over rows
  for(const dolfin::la_index* i_ptr = rows; i_ptr != rows + m; ++i_ptr)
  {
    // Does this row have diagonal?
    diagonal_unset = *i_ptr <= num_cols && *i_ptr >= 0;

    // Loop over non-zeros in a row
    for (eigen_matrix_type::InnerIterator it(_matA, *i_ptr); it; ++it)
    {
      // Check if we are on the diagonal
      if (diagonal_unset && it.index() == *i_ptr)
      {
        it.valueRef() = 1.0;
        diagonal_unset = false;
      }
      else
        it.valueRef() = 0.0;
    }

    // Check that diagonal has been set
    if (diagonal_unset)
    {
      dolfin_error("EigenMatrix.cpp",
                   "set rows to identity",
                   "Diagonal element at row %d not preallocated. "
                   "Use assembler option keep_diagonal", *i_ptr);
    }
  }
}
//---------------------------------------------------------------------------
void EigenMatrix::mult(const GenericVector& x, GenericVector& y) const
{
  const EigenVector& xx = as_type<const EigenVector>(x);
  EigenVector& yy = as_type<EigenVector>(y);
  if (size(1) != xx.size())
  {
    dolfin_error("EigenMatrix.cpp",
                 "compute matrix-vector product with Eigen matrix",
                 "Non-matching dimensions for matrix-vector product");
  }

  // Resize RHS if empty
  if (yy.empty())
    init_vector(yy, 0);

  if (size(0) != yy.size())
  {
    dolfin_error("EigenMatrix.cpp",
                 "compute matrix-vector product with Eigen matrix",
                 "Vector for matrix-vector result has wrong size");
  }

  yy.vec() = _matA*xx.vec();
}
//-----------------------------------------------------------------------------
void EigenMatrix::get_diagonal(GenericVector& x) const
{
  if (size(1) != size(0) || size(0) != x.size())
  {
    dolfin_error("EigenMatrix.cpp",
                 "Get diagonal of a Eigen Matrix",
                 "Matrix and vector dimensions don't match");
  }

  Eigen::VectorXd& xx = x.down_cast<EigenVector>().vec();
  for (std::size_t i = 0; i != x.size(); ++i)
    xx[i] = _matA.coeff(i, i);
}
//-----------------------------------------------------------------------------
void EigenMatrix::set_diagonal(const GenericVector& x)
{
  if (size(1) != size(0) || size(0) != x.size())
  {
    dolfin_error("EigenMatrix.cpp",
                 "Set diagonal of a Eigen Matrix",
                 "Matrix and vector dimensions don't match");
  }

  const Eigen::VectorXd& xx = x.down_cast<EigenVector>().vec();
  for (std::size_t i = 0; i != x.size(); ++i)
    _matA.coeffRef(i, i) = xx[i];
}
//----------------------------------------------------------------------------
void EigenMatrix::transpmult(const GenericVector& x,
                             GenericVector& y) const
{
  const EigenVector& xx = as_type<const EigenVector>(x);
  EigenVector& yy = as_type<EigenVector>(y);

  if (size(0) != xx.size())
  {
    dolfin_error("EigenMatrix.cpp",
                 "compute matrix-vector product with Eigen matrix",
                 "Non-matching dimensions for matrix-vector product");
  }

  // Resize RHS if empty
  if (yy.empty())
    init_vector(yy, 1);

  if (size(1) != yy.size())
  {
    dolfin_error("EigenMatrix.cpp",
                 "compute matrix-vector product with Eigen matrix",
                 "Vector for matrix-vector result has wrong size");
  }

  yy.vec() = _matA.transpose()*xx.vec();
}
//----------------------------------------------------------------------------
const EigenMatrix& EigenMatrix::operator*= (double a)
{
  _matA *= a;
  return *this;
}
//----------------------------------------------------------------------------
const EigenMatrix& EigenMatrix::operator/= (double a)
{
  _matA /= a;
  return *this;
}
//----------------------------------------------------------------------------
const GenericMatrix& EigenMatrix::operator= (const GenericMatrix& A)
{
  *this = as_type<const EigenMatrix>(A);
  return *this;
}
//----------------------------------------------------------------------------
const EigenMatrix& EigenMatrix::operator= (const EigenMatrix& A)
{
  // Check for self-assignment
  if (this != &A)
    _matA = A.mat();

  return *this;
}
//----------------------------------------------------------------------------
std::tuple<const int*, const int*, const double*, std::size_t>
EigenMatrix:: data() const
{
  // Check that matrix has been compressed
  if (!_matA.isCompressed())
  {
    dolfin_error("EigenMatrix.cpp",
                 "get raw data from EigenMatrix",
                 "Matrix has not been compressed. Try calling EigenMatrix::compress() first");
  }

  // Return pointers to matrix data
  return std::make_tuple(_matA.outerIndexPtr(), _matA.innerIndexPtr(),
                         _matA.valuePtr(), _matA.nonZeros());
}
//----------------------------------------------------------------------------
std::string EigenMatrix::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    for (dolfin::la_index it1 = 0; it1 != _matA.outerSize(); ++it1)
    {
      s << "|";
      for (eigen_matrix_type::InnerIterator it2(_matA, it1); it2; ++it2)
      {
        std::stringstream entry;
        entry << std::setiosflags(std::ios::scientific);
        entry << std::setprecision(16);
        entry << " (" << it2.row() << ", " << it2.col() << ", " << it2.value()
              << ")";
        s << entry.str();
      }
      s << " |" << std::endl;
    }
  }
  else
    s << "<EigenMatrix of size " << size(0) << " x " << size(1) << ">";

  return s.str();
}
//----------------------------------------------------------------------------
void EigenMatrix::init(const TensorLayout& tensor_layout)
{
  resize(tensor_layout.size(0), tensor_layout.size(1));

  // Get sparsity pattern
  dolfin_assert(tensor_layout.sparsity_pattern());
  auto sparsity_pattern = tensor_layout.sparsity_pattern();
  dolfin_assert(sparsity_pattern);

  // Reserve space for non-zeroes and get non-zero pattern
  std::vector<std::size_t> num_nonzeros_per_row;
  sparsity_pattern->num_nonzeros_diagonal(num_nonzeros_per_row);
  _matA.reserve(num_nonzeros_per_row);

  const std::vector<std::vector<std::size_t>> pattern
    = sparsity_pattern->diagonal_pattern(SparsityPattern::sorted);

  if (!eigen_matrix_type::IsRowMajor)
    warning ("Entering sparsity for RowMajor matrix - performance may be affected");

  // Add entries for RowMajor matrix
  for (std::size_t i = 0; i != pattern.size(); ++i)
  {
    for (auto j : pattern[i])
      _matA.insert(i, j) = 0.0;
  }
}
//---------------------------------------------------------------------------
std::size_t EigenMatrix::nnz() const
{
  return _matA.nonZeros();
}
//---------------------------------------------------------------------------
void EigenMatrix::apply(std::string mode)
{
  _matA.makeCompressed();
}
//---------------------------------------------------------------------------
void EigenMatrix::axpy(double a, const GenericMatrix& A,
                       bool same_nonzero_pattern)
{
  // Check for same size
  if (size(0) != A.size(0) or size(1) != A.size(1))
  {
    dolfin_error("EigenMatrix.cpp",
                 "perform axpy operation with Eigen matrix",
                 "Dimensions don't match");
  }

  _matA += (a)*(as_type<const EigenMatrix>(A).mat());
}
//-----------------------------------------------------------------------------
