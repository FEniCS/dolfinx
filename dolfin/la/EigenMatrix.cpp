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

#include "EigenMatrix.h"
#include "EigenFactory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& EigenMatrix::factory() const
{
  return EigenFactory::instance();
}
//---------------------------------------------------------------------------
EigenMatrix::EigenMatrix() : GenericMatrix(), _matA(0, 0)
{
// Do nothing
}
//---------------------------------------------------------------------------
EigenMatrix::EigenMatrix(std::size_t M, std::size_t N)
  : GenericMatrix(), _matA(M, N)
{
  // Do nothing
}
//---------------------------------------------------------------------------
EigenMatrix::EigenMatrix(const EigenMatrix& A)
  : GenericMatrix(), _matA(A._matA)
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
  std::shared_ptr<GenericMatrix> A(new EigenMatrix(*this));
  return A;
}
//---------------------------------------------------------------------------
void EigenMatrix::resize(std::size_t M, std::size_t N)
{
  // Resize matrix
  if( size(0) != M || size(1) != N )
    _matA.resize(M, N);
}
//---------------------------------------------------------------------------
std::size_t EigenMatrix::size(std::size_t dim) const
{
  if (dim > 1)
  {
    dolfin_error("EigenMatrix.h",
                 "access size of Eigen matrix",
                 "Illegal axis (%d), must be 0 or 1", dim);
  }

  dolfin_assert(dim < 2);
  return (dim == 0 ? _matA.rows() : _matA.cols());
}
//---------------------------------------------------------------------------
double EigenMatrix::norm(std::string norm_type) const
{
  if (norm_type == "l2")
    return _matA.squaredNorm();
  //  else if (norm_type == "l1")
  //    return _matA.lpNorm<1>();
  //  else if (norm_type == "linf")
  //    return _matA.lpNorm<Eigen::Infinity>();
  else if (norm_type == "frobenius")
    return _matA.norm();
  else
  {
  dolfin_error("EigenMatrix.h",
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

  // Insert values into std::vectors
  columns.clear();
  values.clear();

  // This works because the storage is Eigen::RowMajor
  for (eigen_matrix_type::InnerIterator
         it(_matA, row_idx); it; ++it)
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
  for (std::size_t i = 0; i < m; i++)
    for (std::size_t j = 0; j < n; j++)
      _matA.coeffRef(rows[i] , cols[j]) = block[i*n + j];
}
//---------------------------------------------------------------------------
void EigenMatrix::add(const double* block, std::size_t m,
                      const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols)
{
  for (std::size_t j = 0; j < n; ++j)
  {
    const dolfin::la_index col = cols[j];
    for (std::size_t i = 0; i < m; ++i)
       _matA.coeffRef(rows[i] , col) += block[i*n + j];
  }
}
//---------------------------------------------------------------------------
void EigenMatrix::get(double* block, std::size_t m,
                      const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols) const
{
  for(std::size_t i = 0; i < m; ++i)
    for(std::size_t j = 0; j < n; ++j)
      block[i*n + j] = _matA.coeff(rows[i], cols[j]);
}
//---------------------------------------------------------------------------
// void EigenMatrix::lump(EigenVector& m) const
// {
//   const std::size_t n = size(1);
//   m.init(mpi_comm(), n);
//   m.zero();
//   ublas::scalar_vector<double> one(n, 1.0);
//   ublas::axpy_prod(_matA, one, m.vec(), true);
// }
// //----------------------------------------------------------------------------
// template <typename Mat>
// void EigenMatrix<Mat>::solve(EigenVector& x, const EigenVector& b) const
// {
//   // Make copy of matrix and vector
//   EigenMatrix<Mat> Atemp;
//   Atemp.mat().resize(size(0), size(1));
//   Atemp.mat().assign(_matA);
//   x.vec().resize(b.vec().size());
//   x.vec().assign(b.vec());

//   // Solve
//   Atemp.solve_in_place(x.vec());
// }
// //----------------------------------------------------------------------------
// template <typename Mat>
// void EigenMatrix<Mat>::solve_in_place(EigenVector& x, const EigenVector& b)
// {
//   const std::size_t M = _matA.size1();
//   dolfin_assert(M == b.size());

//   // Initialise solution vector
//   if( x.vec().size() != M )
//     x.vec().resize(M);
//   x.vec().assign(b.vec());

//   // Solve
//   solve_in_place(x.vec());
// }
// //----------------------------------------------------------------------------
// template <typename Mat>
// void EigenMatrix<Mat>::invert()
// {
//   const std::size_t M = _matA.size1();
//   dolfin_assert(M == _matA.size2());

//   // Create identity matrix
//   Mat X(M, M);
//   X.assign(ublas::identity_matrix<double>(M));

//   // Solve
//   solve_in_place(X);
//   _matA.assign_temporary(X);
// }
//---------------------------------------------------------------------------
void EigenMatrix::zero()
{
  // Set to zero whilst keeping the non-zero pattern
  _matA *= 0.0;
}
//----------------------------------------------------------------------------
void EigenMatrix::zero(std::size_t m, const dolfin::la_index* rows)
{
  for(const dolfin::la_index* ptr = rows; ptr != rows + m; ++ptr)
  {
    _matA.row(*ptr) *= 0.0;
  }
}
//----------------------------------------------------------------------------
void EigenMatrix::ident(std::size_t m, const dolfin::la_index* rows)
{
  for(const dolfin::la_index* ptr = rows; ptr != rows + m; ++ptr)
  {
    _matA.row(*ptr) *= 0.0;
    _matA.coeffRef(*ptr, *ptr) = 1.0;
  }
}
//---------------------------------------------------------------------------
void EigenMatrix::mult(const GenericVector& x, GenericVector& y) const
{
  const EigenVector& xx = as_type<const EigenVector>(x);
  EigenVector& yy = as_type<EigenVector>(y);

  if (size(1) != xx.size())
  {
    dolfin_error("EigenMatrix.h",
                 "compute matrix-vector product with Eigen matrix",
                 "Non-matching dimensions for matrix-vector product");
  }

  // Resize RHS if empty
  if (yy.empty())
    init_vector(yy, 0);

  if (size(0) != yy.size())
  {
    dolfin_error("EigenMatrix.h",
                 "compute matrix-vector product with Eigen matrix",
                 "Vector for matrix-vector result has wrong size");
  }

  yy.vec() = _matA*xx.vec();
}
//-----------------------------------------------------------------------------
void EigenMatrix::set_diagonal(const GenericVector& x)
{
  if (size(1) != size(0) || size(0) != x.size())
  {
    dolfin_error("EigenMatrix.h",
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
    dolfin_error("EigenMatrix.h",
                 "compute matrix-vector product with Eigen matrix",
                 "Non-matching dimensions for matrix-vector product");
  }

  // Resize RHS if empty
  if (yy.empty())
    init_vector(yy, 1);

  if (size(1) != yy.size())
  {
    dolfin_error("EigenMatrix.h",
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
const
EigenMatrix& EigenMatrix::operator= (const EigenMatrix& A)
{
  // Check for self-assignment
  if (this != &A)
  {
    _matA = A.mat();
  }
  return *this;
}
//----------------------------------------------------------------------------
// void EigenMatrix::compress()
// {
//   Mat A_temp(this->size(0), this->size(1));
//   A_temp.assign(_matA);
//   _matA.swap(A_temp);
// }
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
        entry << " (" << it2.row() << ", " << it2.col() << ", " << it2.value() << ")";
        s << entry.str();
      }
      s << " |" << std::endl;
    }
  }
  else
  {
    s << "<EigenMatrix of size " << size(0) << " x " << size(1) << ">";
  }

  return s.str();
}
//----------------------------------------------------------------------------
void
EigenMatrix::init(const TensorLayout& tensor_layout)
{
  resize(tensor_layout.size(0), tensor_layout.size(1));

  // Get sparsity pattern
  dolfin_assert(tensor_layout.sparsity_pattern());
  const SparsityPattern* pattern_pointer
    = dynamic_cast<const SparsityPattern*>(tensor_layout.sparsity_pattern().get());
  if (!pattern_pointer)
  {
    dolfin_error("EigenMatrix.h",
                 "initialize Eigen matrix",
                 "Cannot convert GenericSparsityPattern to concrete SparsityPattern type");
  }

  // Reserve space for non-zeroes and get non-zero pattern
  std::vector<std::size_t> num_nonzeros_per_row;
  pattern_pointer->num_nonzeros_diagonal(num_nonzeros_per_row);
  _matA.reserve(num_nonzeros_per_row);

  const std::vector<std::vector<std::size_t> > pattern
    = pattern_pointer->diagonal_pattern(SparsityPattern::sorted);

  // Add entries for RowMajor matrix
  for (std::size_t i = 0; i != pattern.size(); ++i)
  {
    for (const auto &j : pattern[i])
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
  // Make sure matrix assembly is complete
}
//---------------------------------------------------------------------------
void EigenMatrix::axpy(double a, const GenericMatrix& A,
                              bool same_nonzero_pattern)
{
  // Check for same size
  if (size(0) != A.size(0) or size(1) != A.size(1))
  {
    dolfin_error("EigenMatrix.h",
                 "perform axpy operation with Eigen matrix",
                 "Dimensions don't match");
  }

  _matA += (a)*(as_type<const EigenMatrix>(A).mat());
}
//-----------------------------------------------------------------------------
