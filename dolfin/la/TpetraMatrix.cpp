// Copyright (C) 2014
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
// First added:  2014

#ifdef HAS_TRILINOS

#include <iomanip>
#include <iostream>
#include <sstream>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/MPI.h>
#include "TpetraVector.h"
#include "TpetraMatrix.h"
#include "GenericSparsityPattern.h"
#include "SparsityPattern.h"
#include "TensorLayout.h"
#include "TpetraFactory.h"

using namespace dolfin;

//const std::map<std::string, NormType> TpetraMatrix::norm_types
//= { {"l1",        NORM_1},
//    {"linf",      NORM_INFINITY},
//    {"frobenius", NORM_FROBENIUS} };

//-----------------------------------------------------------------------------
TpetraMatrix::TpetraMatrix() : _matA(NULL)
{
  // Do nothing else
}
//-----------------------------------------------------------------------------
TpetraMatrix::TpetraMatrix(Teuchos::RCP<matrix_type> A) : _matA(A)
{
}
//-----------------------------------------------------------------------------
TpetraMatrix::TpetraMatrix(const TpetraMatrix& A)
{
  if (!A._matA.is_null())
  {
  dolfin_not_implemented();
  }
}
//-----------------------------------------------------------------------------
TpetraMatrix::~TpetraMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> TpetraMatrix::copy() const
{
  return std::shared_ptr<GenericMatrix>(new TpetraMatrix(*this));
}
//-----------------------------------------------------------------------------
void TpetraMatrix::init(const TensorLayout& tensor_layout)
{
  if (!_matA.is_null())
    error("TpetraMatrix may not be initialized more than once.");

  // Get global dimensions and local range
  dolfin_assert(tensor_layout.rank() == 2);
  const std::size_t M = tensor_layout.size(0);
  const std::size_t N = tensor_layout.size(1);
  const std::pair<std::size_t, std::size_t> row_range
    = tensor_layout.local_range(0);
  const std::pair<std::size_t, std::size_t> col_range
    = tensor_layout.local_range(1);
  const std::size_t m = row_range.second - row_range.first;
  const std::size_t n = col_range.second - col_range.first;

  // Get sparsity pattern
  dolfin_assert(tensor_layout.sparsity_pattern());
  const GenericSparsityPattern& sparsity_pattern
    = *tensor_layout.sparsity_pattern();

  std::cout << "num_non_zeros = " << sparsity_pattern.num_nonzeros() <<" \n";

  // Initialize matrix
  // Insist on square Matrix for now
  dolfin_assert(M == N);

  Teuchos::RCP<const Teuchos::Comm<int> >
    _comm(new Teuchos::MpiComm<int>(sparsity_pattern.mpi_comm()));

  std::cout << _comm->getRank() << " : " << row_range.first << "-" << row_range.second << "\n";

  Teuchos::RCP<const map_type> row_map(new map_type(M, m, 0, _comm));
  Teuchos::RCP<const map_type> col_map(new map_type(N, n, 0, _comm));

  //  typedef Tpetra::CrsGraph<> graph_type;
  //
  // // Make a Tpetra::CrsGraph of the sparsity_pattern
  // // Get the number of entries on each row
  // std::vector<std::size_t> num_local_nz;
  // sparsity_pattern.num_local_nonzeros(num_local_nz);

  // const Teuchos::ArrayRCP<const std::size_t> _entries_per_row(num_local_nz.size(), 10);
  // Teuchos::RCP<graph_type> _graph(new graph_type(row_map, _entries_per_row));
  // std::pair<std::size_t, std::size_t> range = sparsity_pattern.local_range(0);
  // std::vector<std::vector<std::size_t> > pattern = sparsity_pattern.off_diagonal_pattern(GenericSparsityPattern::unsorted);

  // std::vector<global_ordinal_type> indices;
  // for (std::size_t i = 0; i != range.second - range.first; ++i)
  // {
  //   indices.clear();
  //   std::copy(pattern[i].begin(), pattern[i].end(), indices.begin());

  //   Teuchos::ArrayView<global_ordinal_type> _indices(indices);
  //   _graph->insertGlobalIndices(range.first + i, _indices);
  // }
  // _graph->fillComplete();

  _matA = Teuchos::rcp(new matrix_type(row_map, col_map, 0));
  //  _matA = Teuchos::rcp(new matrix_type(_graph));

}
//-----------------------------------------------------------------------------
std::size_t TpetraMatrix::size(std::size_t dim) const
{
  if (dim > 1)
  {
    dolfin_error("TpetraMatrix.cpp",
                 "access size of Tpetra matrix",
                 "Illegal axis (%d), must be 0 or 1", dim);
  }

  std::size_t num_elements;
  if (_matA.is_null())
    num_elements = 0;
  else if (dim == 0)
    num_elements = _matA->getRowMap()->getGlobalNumElements();
  else
    num_elements = _matA->getColMap()->getGlobalNumElements();

  return num_elements;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t> TpetraMatrix::local_range(std::size_t dim) const
{
  std::cout << "TpetraMatrix::local_range()\n";
  
  return std::make_pair(0,0);
}
//-----------------------------------------------------------------------------
bool TpetraMatrix::empty() const
{
  return _matA.is_null();
}
//-----------------------------------------------------------------------------
void TpetraMatrix::init_vector(GenericVector& z, std::size_t dim) const
{
  dolfin_assert(!_matA.is_null());

  // Downcast vector
  TpetraVector& _z = as_type<TpetraVector>(z);

  Teuchos::RCP<const map_type> _map;

  if (dim == 0)
  {
    _map = _matA->getRowMap();
  }
  else if (dim == 1)
  {
    _map = _matA->getColMap();
  }
  else
  {
    dolfin_error("TpetraMatrix.cpp",
                 "initialize Tpetra vector to match Tpetra matrix",
                 "Dimension must be 0 or 1, not %d", dim);
  }

  _z._x = Teuchos::rcp(new vector_type(_map));

}
//-----------------------------------------------------------------------------
void TpetraMatrix::get(double* block,
                      std::size_t m, const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols) const
{
  // Get matrix entries (must be on this process)
  dolfin_assert(!_matA.is_null());

  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const global_ordinal_type> _columns;
    Teuchos::ArrayView<const scalar_type> _data;
    _matA->getGlobalRowView(rows[i], _columns, _data);
    // FIXME: get desired columns from all columns
  }
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void TpetraMatrix::set(const double* block,
                      std::size_t m, const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(!_matA.is_null());

  std::cout << "Set " << m << ", " << n << "\n";

  // Tpetra View of column indices
  Teuchos::ArrayView<const global_ordinal_type> column_idx(cols, n);

  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const scalar_type> data(block + i*n, n);
    _matA->insertGlobalValues(rows[i], column_idx, data);
  }

}
//-----------------------------------------------------------------------------
void TpetraMatrix::set_local(const double* block,
                            std::size_t m, const dolfin::la_index* rows,
                            std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(!_matA.is_null());

  std::cout << "Set local  " << m << ", " << n << "\n";

  // Tpetra View of column indices
  Teuchos::ArrayView<const local_ordinal_type> column_idx(cols, n);

  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const scalar_type> data(block + i*n, n);
    _matA->insertLocalValues(rows[i], column_idx, data);
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::add(const double* block,
                      std::size_t m, const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(!_matA.is_null());

  // FIXME: add is the same as set, because using insert*Values to
  // insert or add entries is needed to create the matrix entry in the first place.
  // Really, should initialise the sparsity pattern first as a CrsGraph, and use 
  // replace*Values and sumInto*Values

  std::cout << "Add  " << m << ", " << n << "\n";
  // Tpetra View of column indices
  Teuchos::ArrayView<const global_ordinal_type> column_idx(cols, n);

  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const scalar_type> data(block + i*n, n);
    _matA->insertGlobalValues(rows[i], column_idx, data);
  }

}
//-----------------------------------------------------------------------------
void TpetraMatrix::add_local(const double* block,
                            std::size_t m, const dolfin::la_index* rows,
                            std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(!_matA.is_null());

  std::stringstream ss;
  ss << "add_local on " << _matA->getMap()->getComm()->getRank() << " [";
  for (std::size_t i = 0; i != m; ++i)
    ss << rows[i] << " ";
  ss << "]\n";

  std::cout << ss.str();

  // Tpetra View of column indices
  Teuchos::ArrayView<const local_ordinal_type> column_idx(cols, n);

  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const scalar_type> data(block + i*n, n);
    if (_matA->getRowMap()->isNodeLocalElement(rows[i]))
      _matA->insertLocalValues(rows[i], column_idx, data);
  }

}
//-----------------------------------------------------------------------------
void TpetraMatrix::axpy(double a, const GenericMatrix& A,
                       bool same_nonzero_pattern)
{
  const TpetraMatrix& AA = as_type<const TpetraMatrix>(A);
  dolfin_assert(!_matA.is_null());
  dolfin_assert(!AA._matA.is_null());

  // Make const matrix result (cannot add in place)
  const Teuchos::RCP<const matrix_type> _matB
    = Teuchos::rcp_dynamic_cast<const matrix_type>(_matA->add(1.0, *AA._matA, a,
                                                   Teuchos::null, Teuchos::null, Teuchos::null));

  std::cout << _matB->description() << "\n";

  // FIXME: copy result in _matB back into _matA

  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void TpetraMatrix::getrow(std::size_t row, std::vector<std::size_t>& columns,
                         std::vector<double>& values) const
{
  dolfin_assert(!_matA.is_null());

  const std::size_t ncols = _matA->getNumEntriesInGlobalRow(row);
  columns.resize(ncols);
  values.resize(ncols);

  // Make tmp vector to match type
  std::vector<global_ordinal_type> columns_tmp(ncols);
  Teuchos::ArrayView<global_ordinal_type> _columns(columns_tmp);
  Teuchos::ArrayView<scalar_type> _values(values);

  std::size_t n;
  _matA->getGlobalRowCopy(row, _columns, _values, n);
  dolfin_assert(n == ncols);

  std::copy(columns_tmp.begin(), columns_tmp.end(), columns.begin());
}
//-----------------------------------------------------------------------------
void TpetraMatrix::setrow(std::size_t row,
                         const std::vector<std::size_t>& columns,
                         const std::vector<double>& values)
{
  dolfin_assert(!_matA.is_null());

  dolfin_not_implemented();

  if (columns.size() != values.size())
  {
    dolfin_error("TpetraMatrix.cpp",
                 "set row of values for Tpetra matrix",
                 "Number of columns and values don't match");
  }

  // Handle case n = 0
  if (columns.size() == 0)
    return;

  // Tpetra View of column indices - copy to get correct type
  std::vector<global_ordinal_type> cols_tmp(columns.begin(), columns.end());
  Teuchos::ArrayView<const global_ordinal_type> column_idx(cols_tmp);

  // Tpetra View of values
  Teuchos::ArrayView<const scalar_type> data(values);

  _matA->insertGlobalValues(row, column_idx, data);
}
//-----------------------------------------------------------------------------
void TpetraMatrix::zero(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(!_matA.is_null());

  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const global_ordinal_type> cols;
    Teuchos::ArrayView<const scalar_type> data;
    _matA->getGlobalRowView(rows[i], cols, data);
    std::vector<double> z(cols.size(), 0);
    Teuchos::ArrayView<const scalar_type> dataz(z);

    _matA->replaceGlobalValues(rows[i], cols, dataz);
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::zero_local(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(!_matA.is_null());
  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const global_ordinal_type> cols;
    Teuchos::ArrayView<const scalar_type> data;
    _matA->getLocalRowView(rows[i], cols, data);
    std::vector<double> z(cols.size(), 0);
    Teuchos::ArrayView<const scalar_type> dataz(z);

    _matA->replaceLocalValues(rows[i], cols, dataz);
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::ident(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(!_matA.is_null());

  // Clear affected rows to zero
  zero(m, rows);

  // Get map of locally available columns
  Teuchos::RCP<const map_type>colmap(_matA->getColMap());

  const scalar_type one = 1;
  Teuchos::ArrayView<const scalar_type> data(&one, 1);
  global_ordinal_type col;
  Teuchos::ArrayView<global_ordinal_type> column_idx(&col, 1);

  // Set diagonal entries where possible
  for (std::size_t i = 0 ; i != m; ++i)
  {
    if (colmap->isNodeGlobalElement(rows[i]))
    {
      col = rows[i];
      _matA->insertGlobalValues(rows[i], column_idx, data);
    }
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::ident_local(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(!_matA.is_null());
  zero_local(m, rows);
  Teuchos::RCP<const map_type>colmap(_matA->getColMap());
  const scalar_type one = 1;
  Teuchos::ArrayView<const scalar_type> data(&one, 1);
  local_ordinal_type col;
  Teuchos::ArrayView<local_ordinal_type> column_idx(&col, 1);
  for (std::size_t i = 0 ; i != m; ++i)
  {
    if (colmap->isNodeLocalElement(rows[i]))
    {
      col = rows[i];
      _matA->insertLocalValues(rows[i], column_idx, data);
    }
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::mult(const GenericVector& x, GenericVector& y) const
{
  dolfin_assert(!_matA.is_null());

  const TpetraVector& xx = as_type<const TpetraVector>(x);
  TpetraVector& yy = as_type<TpetraVector>(y);

  if (size(1) != xx.size())
  {
    dolfin_error("TpetraMatrix.cpp",
                 "compute matrix-vector product with Tpetra matrix",
                 "Non-matching dimensions for matrix-vector product");
  }

  // Resize RHS if empty
  if (yy.size() == 0)
    init_vector(yy, 0);

  if (size(0) != yy.size())
  {
    dolfin_error("TpetraMatrix.cpp",
                 "compute matrix-vector product with Tpetra matrix",
                 "Vector for matrix-vector result has wrong size");
  }

  _matA->apply(*xx._x, *yy._x);
}
//-----------------------------------------------------------------------------
void TpetraMatrix::transpmult(const GenericVector& x, GenericVector& y) const
{
  dolfin_assert(!_matA.is_null());

  const TpetraVector& xx = as_type<const TpetraVector>(x);
  TpetraVector& yy = as_type<TpetraVector>(y);

  if (size(0) != xx.size())
  {
    dolfin_error("TpetraMatrix.cpp",
                 "compute transpose matrix-vector product with Tpetra matrix",
                 "Non-matching dimensions for transpose matrix-vector product");
  }

  // Resize RHS if empty
  if (yy.size() == 0)
    init_vector(yy, 1);

  if (size(1) != yy.size())
  {
    dolfin_error("TpetraMatrix.cpp",
                 "compute transpose matrix-vector product with Tpetra matrix",
                 "Vector for transpose matrix-vector result has wrong size");
  }

  _matA->apply(*xx._x, *yy._x, Teuchos::TRANS);

}
//-----------------------------------------------------------------------------
void TpetraMatrix::set_diagonal(const GenericVector& x)
{
  dolfin_assert(!_matA.is_null());
  dolfin_not_implemented();

  const TpetraVector& xx = x.down_cast<TpetraVector>();
  if (size(1) != size(0) || size(0) != xx.size())
  {
    dolfin_error("TpetraMatrix.cpp",
                 "set diagonal of a Tpetra matrix",
                 "Matrix and vector dimensions don't match for matrix-vector set");
  }

  apply("insert");
}
//-----------------------------------------------------------------------------
double TpetraMatrix::norm(std::string norm_type) const
{
  dolfin_assert(!_matA.is_null());
  dolfin_not_implemented();

  // Check that norm is known
  // if (norm_types.count(norm_type) == 0)
  // {
  //   dolfin_error("TpetraMatrix.cpp",
  //                "compute norm of Tpetra matrix",
  //                "Unknown norm type (\"%s\")", norm_type.c_str());
  // }

  double value = 0.0;

  return value;
}
//-----------------------------------------------------------------------------
void TpetraMatrix::apply(std::string mode)
{
  Timer timer("Apply (TpetraMatrix)");

  dolfin_assert(!_matA.is_null());

  if (mode == "add" or mode == "insert")
    _matA->fillComplete();
  else
    dolfin_error("TpetraMatrix.cpp",
                 "apply changes to Tpetra matrix",
                 "Unknown apply mode \"%s\"", mode.c_str());
}
//-----------------------------------------------------------------------------
MPI_Comm TpetraMatrix::mpi_comm() const
{
  // Unwrap MPI_Comm
  const Teuchos::RCP<const Teuchos::MpiComm<int> > _mpi_comm
    = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >
    (_matA->getMap()->getComm());

  return *(_mpi_comm->getRawMpiComm());
}
//-----------------------------------------------------------------------------
void TpetraMatrix::zero()
{
  dolfin_assert(!_matA.is_null());
  _matA->setAllToScalar(0.0);
}
//-----------------------------------------------------------------------------
const TpetraMatrix& TpetraMatrix::operator*= (double a)
{
  dolfin_assert(!_matA.is_null());
  _matA->scale(a);
  return *this;
}
//-----------------------------------------------------------------------------
const TpetraMatrix& TpetraMatrix::operator/= (double a)
{
  dolfin_assert(!_matA.is_null());
  _matA->scale(1.0/a);
  return *this;
}
//-----------------------------------------------------------------------------
const GenericMatrix& TpetraMatrix::operator= (const GenericMatrix& A)
{
  *this = as_type<const TpetraMatrix>(A);
  return *this;
}
//-----------------------------------------------------------------------------
bool TpetraMatrix::is_symmetric(double tol) const
{
  dolfin_assert(!_matA.is_null());
  return false;
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& TpetraMatrix::factory() const
{
  return TpetraFactory::instance();
}
//-----------------------------------------------------------------------------
const TpetraMatrix& TpetraMatrix::operator= (const TpetraMatrix& A)
{
  return *this;
}
//-----------------------------------------------------------------------------
std::string TpetraMatrix::str(bool verbose) const
{
  if (_matA.is_null())
    return "<Uninitialized TpetraMatrix>";

  std::stringstream s;
  if (verbose)
  {
    s << "< " << _matA->description() << " >";
  }
  else
    s << "<TpetraMatrix of size " << size(0) << " x " << size(1) << ">";

  return s.str();
}
//-----------------------------------------------------------------------------

#endif
