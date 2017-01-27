// Copyright (C) 2014 Chris Richardson
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
// First added: 2014-12-20

#ifdef HAS_TRILINOS

#include <iomanip>
#include <iostream>
#include <sstream>

#include <dolfin/log/log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/MPI.h>
#include "TpetraVector.h"
#include "TpetraMatrix.h"
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
  // Do nothing
}
//-----------------------------------------------------------------------------
TpetraMatrix::TpetraMatrix(Teuchos::RCP<matrix_type> A) : _matA(A)
{
  // row_map and col_map are not set, so cannot use add_local() or set_local()
}
//-----------------------------------------------------------------------------
TpetraMatrix::TpetraMatrix(const TpetraMatrix& A)
{
  // Copy NULL to NULL
  if(A._matA.is_null())
    return;

  _matA = A._matA->clone(A._matA->getNode());

  index_map[0] = A.index_map[0];
  index_map[1] = A.index_map[1];

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
  {
    dolfin_error("TpetraMatrix.h",
                 "initialize matrix",
                 "Matrix cannot be initialised more than once");
  }

  // Get global dimensions and local range
  dolfin_assert(tensor_layout.rank() == 2);

  const std::pair<std::int64_t, std::int64_t> row_range
    = tensor_layout.local_range(0);
  const std::size_t m = row_range.second - row_range.first;

  const std::pair<std::int64_t, std::int64_t> col_range
    = tensor_layout.local_range(1);
  const std::size_t n = col_range.second - col_range.first;

  // Get sparsity pattern
  auto sparsity_pattern = tensor_layout.sparsity_pattern();
  dolfin_assert(sparsity_pattern);

  // Initialize matrix

  // Set up MPI Comm
  Teuchos::RCP<const Teuchos::Comm<int>>
    _comm(new Teuchos::MpiComm<int>(sparsity_pattern->mpi_comm()));

  // Save the local row and column mapping, so we can use add_local
  // later with off-process entries

  // Overlapping RowMap
  index_map[0] = tensor_layout.index_map(0);

  std::vector<dolfin::la_index> global_indices0(m);
  for (std::size_t i = 0; i < m; ++i)
  {
    global_indices0[i]
      = tensor_layout.index_map(0)->local_to_global(i);
  }

  // Non-overlapping RangeMap
  Teuchos::ArrayView<dolfin::la_index>
    _global_indices0(global_indices0.data(), m);
  Teuchos::RCP<const map_type> range_map
    (new map_type(Teuchos::OrdinalTraits<dolfin::la_index>::invalid(),
                  _global_indices0, 0, _comm));

  // Overlapping ColMap
  index_map[1] = tensor_layout.index_map(1);

  std::vector<dolfin::la_index> global_indices1(n);
  {
    for (std::size_t i = 0; i < n; ++i)
      global_indices1[i]
        = tensor_layout.index_map(1)->local_to_global(i);
  }

  // Non-overlapping DomainMap
  Teuchos::ArrayView<dolfin::la_index>
    _global_indices1(global_indices1.data(), n);
  Teuchos::RCP<const map_type> domain_map
    (new map_type(Teuchos::OrdinalTraits<dolfin::la_index>::invalid(),
                  _global_indices1, 0, _comm));

  std::vector<std::vector<std::size_t>> pattern_diag
    = sparsity_pattern->diagonal_pattern(SparsityPattern::Type::unsorted);
  std::vector<std::vector<std::size_t>> pattern_off
    = sparsity_pattern->off_diagonal_pattern(SparsityPattern::Type::unsorted);
  const bool has_off_diag = pattern_off.size() > 0;

  dolfin_assert(pattern_diag.size() == pattern_off.size() || !has_off_diag);
  dolfin_assert(m == pattern_diag.size());

  // Get number of non-zeros per row to allocate storage
  std::vector<std::size_t> entries_per_row(m);
  sparsity_pattern->num_local_nonzeros(entries_per_row);
  Teuchos::ArrayRCP<std::size_t> _nnz(entries_per_row.data(), 0,
                                      entries_per_row.size(), false);

  // Create a non-overlapping "row" map for the graph
  // The column map will be auto-generated from the entries.

  Teuchos::RCP<graph_type> crs_graph
    (new graph_type(range_map, _nnz, Tpetra::StaticProfile));

  for (std::size_t i = 0; i != m; ++i)
  {
    std::vector<dolfin::la_index> indices(pattern_diag[i].begin(),
                                          pattern_diag[i].end());
    if (has_off_diag)
    {
      indices.insert(indices.end(), pattern_off[i].begin(),
                     pattern_off[i].end());
    }

    Teuchos::ArrayView<dolfin::la_index> _indices(indices);
    crs_graph->insertGlobalIndices
      ((dolfin::la_index)tensor_layout.index_map(0)->local_to_global(i), _indices);
  }

  crs_graph->fillComplete(domain_map, range_map);

  _matA = Teuchos::rcp(new matrix_type(crs_graph));
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
    num_elements = _matA->getRowMap()->getMaxAllGlobalIndex() + 1;
  else
    num_elements = _matA->getColMap()->getMaxAllGlobalIndex() + 1;

  return num_elements;
}
//-----------------------------------------------------------------------------
std::pair<std::int64_t, std::int64_t>
TpetraMatrix::local_range(std::size_t dim) const
{
  dolfin_assert(!_matA.is_null());

  if (dim == 0)
  {
    // Teuchos::RCP<const map_type> a_row_map(_matA->getRowMap());
    // return std::make_pair<std::size_t, std::size_t>
    //   (a_row_map->getMinGlobalIndex(), a_row_map->getMaxGlobalIndex() + 1);

    return index_map[0]->local_range();
  }
  else if (dim == 1)
  {
    // FIXME: this is not quite right - column map will have overlap
    // Teuchos::RCP<const map_type> a_col_map(_matA->getColMap());
    // return std::make_pair<std::size_t, std::size_t>
    //   (a_col_map->getMinGlobalIndex(), a_col_map->getMaxGlobalIndex() + 1);

    return index_map[1]->local_range();
  }
  else
  {
    dolfin_error("TpetraMatrix.cpp",
                 "get local range",
                 "Dimension invalid");
  }

  return std::make_pair(0, 0);
}
//-----------------------------------------------------------------------------
std::size_t TpetraMatrix::nnz() const
{
  dolfin_assert(!_matA.is_null());

  std::size_t nnz_local = _matA->getCrsGraph()->getNodeNumEntries();
  return MPI::sum(mpi_comm(), nnz_local);
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
    _map =_matA->getRangeMap();
    dolfin_assert(_map->isOneToOne());
    _z._x_ghosted = Teuchos::rcp(new TpetraVector::vector_type(_map, 1));
    dolfin_assert(!_z._x_ghosted.is_null());
    // Get a modifiable view into the ghosted vector
    _z._x = _z._x_ghosted->offsetViewNonConst(_map, 0);
  }
  else if (dim == 1)
  {
    _map = _matA->getDomainMap();
    dolfin_assert(_map->isOneToOne());
    _z._x_ghosted = Teuchos::rcp(new TpetraVector::vector_type(_map, 1));
    dolfin_assert(!_z._x_ghosted.is_null());
    // Get a modifiable view into the ghosted vector
    _z._x = _z._x_ghosted->offsetViewNonConst(_map, 0);
  }
  else
  {
    dolfin_error("TpetraMatrix.cpp",
                 "initialize Tpetra vector to match Tpetra matrix",
                 "Dimension must be 0 or 1, not %d", dim);
  }
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
    Teuchos::ArrayView<dolfin::la_index> _columns;
    Teuchos::ArrayView<double> _data;
    std::size_t n;
    _matA->getGlobalRowCopy(rows[i], _columns, _data, n);

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
  dolfin_assert(!_matA->isFillComplete());

  // Tpetra View of column indices
  Teuchos::ArrayView<const dolfin::la_index> column_idx(cols, n);
  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const double> data(block + i*n, n);
    _matA->replaceGlobalValues(rows[i], column_idx, data);
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::set_local(const double* block,
                             std::size_t m, const dolfin::la_index* rows,
                             std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(!_matA.is_null());
  dolfin_assert(!_matA->isFillComplete());

  // Map local columns to global
  std::vector<dolfin::la_index> _global_col_idx;
  _global_col_idx.reserve(n);
  for (std::size_t i = 0 ; i != n; ++i)
    _global_col_idx.push_back(index_map[1]->local_to_global(cols[i]));
  Teuchos::ArrayView<const dolfin::la_index> global_col_idx(_global_col_idx);

  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const double> data(block + i*n, n);

    const dolfin::la_index global_row_idx
      = index_map[0]->local_to_global(rows[i]);
    if (global_row_idx != Teuchos::OrdinalTraits<dolfin::la_index>::invalid())
    {
      std::size_t nvalid = _matA->replaceGlobalValues(global_row_idx,
                                                      global_col_idx, data);
      dolfin_assert(nvalid == n);
    }
    else
      warning("Could not enter into row:%d", rows[i]);
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::add(const double* block,
                       std::size_t m, const dolfin::la_index* rows,
                       std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(!_matA.is_null());
  dolfin_assert(!_matA->isFillComplete());

  // Tpetra View of column indices
  Teuchos::ArrayView<const dolfin::la_index> column_idx(cols, n);
  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const double> data(block + i*n, n);
    _matA->sumIntoGlobalValues(rows[i], column_idx, data);
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::add_local(const double* block,
                             std::size_t m, const dolfin::la_index* rows,
                             std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(!_matA.is_null());
  dolfin_assert(!_matA->isFillComplete());

  // Map local columns to global
  std::vector<dolfin::la_index> _global_col_idx;
  _global_col_idx.reserve(n);
  for (std::size_t i = 0 ; i != n; ++i)
    _global_col_idx.push_back(index_map[1]->local_to_global(cols[i]));
  Teuchos::ArrayView<const dolfin::la_index> global_col_idx(_global_col_idx);

  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const double> data(block + i*n, n);

    const dolfin::la_index global_row_idx
      = index_map[0]->local_to_global(rows[i]);
    if (global_row_idx != Teuchos::OrdinalTraits<dolfin::la_index>::invalid())
    {
      std::size_t nvalid =
        _matA->sumIntoGlobalValues(global_row_idx,
                                   global_col_idx, data);
      dolfin_assert(nvalid == n);
    }
    else
      warning("Could not enter into row:%d", rows[i]);
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::axpy(double a, const GenericMatrix& A,
                        bool same_nonzero_pattern)
{
  dolfin_assert(!_matA.is_null());
  const TpetraMatrix& AA = as_type<const TpetraMatrix>(A);
  dolfin_assert(!AA._matA.is_null());

  double one=1;

  Teuchos::RCP<const matrix_type> matB
    = Teuchos::rcp_dynamic_cast<const matrix_type>
    (_matA->add(a, *AA._matA, one, Teuchos::null,Teuchos::null,Teuchos::null));

  _matA = matB->clone(matB->getNode());

}
//-----------------------------------------------------------------------------
void TpetraMatrix::getrow(std::size_t row, std::vector<std::size_t>& columns,
                         std::vector<double>& values) const
{
  dolfin_assert(!_matA.is_null());

  const std::size_t ncols = _matA->getNumEntriesInGlobalRow(row);
  if (ncols == Teuchos::OrdinalTraits<std::size_t>::invalid())
  {
    dolfin_error("TpetraMatrix.cpp",
                 "get TpetraMatrix row",
                 "Row %d not in range", row);
  }

  columns.resize(ncols);
  values.resize(ncols);

  // Make tmp vector to match type
  std::vector<dolfin::la_index> columns_tmp(ncols);
  Teuchos::ArrayView<dolfin::la_index> _columns(columns_tmp);
  Teuchos::ArrayView<double> _values(values);

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
  dolfin_assert(!_matA->isFillComplete());

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
  std::vector<dolfin::la_index> cols_tmp(columns.begin(), columns.end());
  Teuchos::ArrayView<const dolfin::la_index> column_idx(cols_tmp);

  // Tpetra View of values
  Teuchos::ArrayView<const double> data(values);

  _matA->replaceGlobalValues(row, column_idx, data);
}
//-----------------------------------------------------------------------------
void TpetraMatrix::zero(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(!_matA.is_null());
  dolfin_assert(!_matA->isFillComplete());

  for (std::size_t i = 0 ; i != m; ++i)
  {
    const std::size_t ncols = _matA->getNumEntriesInGlobalRow(rows[i]);
    std::vector<dolfin::la_index> colcols(ncols);
    Teuchos::ArrayView<dolfin::la_index> cols(colcols);
    std::vector<double> coldata(ncols);
    Teuchos::ArrayView<double> data(coldata);
    std::size_t n;
    _matA->getGlobalRowCopy(rows[i], cols, data, n);
    dolfin_assert(n == ncols);
    std::fill(coldata.begin(), coldata.end(), 0.0);
    _matA->replaceGlobalValues(rows[i], cols, data);
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::zero_local(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(!_matA.is_null());
  dolfin_assert(!_matA->isFillComplete());

  for (std::size_t i = 0 ; i != m; ++i)
  {
    int row = rows[i];
    Teuchos::ArrayView<const int> cols;
    Teuchos::ArrayView<const double> data;
    _matA->getLocalRowView(row, cols, data);
    std::vector<double> z(cols.size(), 0);
    Teuchos::ArrayView<const double> dataz(z);

    _matA->replaceLocalValues(row, cols, dataz);
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::ident(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(!_matA.is_null());
  if(_matA->isFillComplete())
    _matA->resumeFill();

  // Clear affected rows to zero
  zero(m, rows);

  // Get map of locally available columns
  Teuchos::RCP<const map_type>colmap(_matA->getColMap());

  const double one = 1;
  Teuchos::ArrayView<const double> data(&one, 1);
  dolfin::la_index col;
  Teuchos::ArrayView<dolfin::la_index> column_idx(&col, 1);

  // Set diagonal entries where possible
  for (std::size_t i = 0 ; i != m; ++i)
  {
    if (colmap->isNodeGlobalElement(rows[i]))
    {
      col = rows[i];
      _matA->replaceGlobalValues(rows[i], column_idx, data);
    }
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::ident_local(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(!_matA.is_null());
  //  dolfin_assert(!_matA->isFillComplete());

  // Will need to call 'apply' again after this
  if(_matA->isFillComplete())
    _matA->resumeFill();

  zero_local(m, rows);

  // FIXME: check this is correct
  Teuchos::RCP<const map_type> colmap(_matA->getColMap());
  const double one = 1.0;
  Teuchos::ArrayView<const double> data(&one, 1);
  int col;
  Teuchos::ArrayView<int> column_idx(&col, 1);

  for (std::size_t i = 0 ; i != m; ++i)
  {
    if (colmap->isNodeLocalElement(rows[i]))
    {
      col = rows[i];
      _matA->replaceLocalValues(rows[i], column_idx, data);
    }
    else
    {
      dolfin_error("TpetraMatrix.cpp",
                   "ident local row",
                   "Row %d not local", rows[i]);
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
                 "Non-matching dimensions %d and %d for matrix-vector product",
                 size(1), xx.size());
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

  _matA->apply(*xx._x, *yy._x_ghosted, Teuchos::TRANS);
}
//-----------------------------------------------------------------------------
void TpetraMatrix::get_diagonal(GenericVector& x) const
{
  dolfin_assert(!_matA.is_null());

  TpetraVector& xx = x.down_cast<TpetraVector>();
  if (!xx._x->getMap()->isSameAs(*_matA->getRangeMap()))
  {
    dolfin_error("TpetraMatrix.cpp",
                 "get diagonal of a Tpetra matrix",
                 "Matrix and vector ColMaps don't match for matrix-vector set");
  }

  if (size(1) != size(0) || size(0) != xx.size())
  {
    dolfin_error("TpetraMatrix.cpp",
                 "get diagonal of a Tpetra matrix",
                 "Matrix and vector dimensions don't match for matrix-vector set");
  }

  if (xx.vec()->getNumVectors() != 1)
  {
    dolfin_error("TpetraMatrix.cpp",
                 "get diagonal of a Tpetra matrix",
                 "Vector is a multivector with %d columns instead of 1", xx.vec()->getNumVectors());
  }

  _matA->getLocalDiagCopy(*(xx._x->getVectorNonConst(0)));

}
//-----------------------------------------------------------------------------
void TpetraMatrix::set_diagonal(const GenericVector& x)
{
  dolfin_assert(!_matA.is_null());
  if(_matA->isFillComplete())
    _matA->resumeFill();

  const TpetraVector& xx = x.down_cast<TpetraVector>();
  if (size(1) != size(0) || size(0) != xx.size())
  {
    dolfin_error("TpetraMatrix.cpp",
                 "set diagonal of a Tpetra matrix",
                 "Matrix and vector dimensions don't match for matrix-vector set");
  }

  dolfin_assert(xx._x->getMap()->isSameAs(*_matA->getRangeMap()));
  Teuchos::ArrayRCP<const double> xarr = xx._x->getData(0);

  dolfin::la_index col_idx;
  double val;
  Teuchos::ArrayView<const dolfin::la_index> global_col_idx(&col_idx, 1);
  Teuchos::ArrayView<const double> data(&val, 1);

  auto range = xx.local_range();
  std::int64_t local_size = range.second - range.first;
  for (std::int64_t i = 0; i < local_size; ++i)
  {
    col_idx = range.first + i;
    val = xarr[i];
    std::size_t nvalid = _matA->replaceGlobalValues(col_idx,
                                                    global_col_idx, data);
    dolfin_assert(nvalid == 1);
  }

  apply("insert");
}
//-----------------------------------------------------------------------------
double TpetraMatrix::norm(std::string norm_type) const
{
  dolfin_assert(!_matA.is_null());

  if (norm_type == "frobenius")
    return _matA->getFrobeniusNorm();
  else
  {
    dolfin_error("TpetraMatrix.cpp",
                 "compute norm of Tpetra matrix",
                 "Unknown norm type (\"%s\")", norm_type.c_str());
  }

  return 0.0;
}
//-----------------------------------------------------------------------------
void TpetraMatrix::apply(std::string mode)
{
  Timer timer("Apply (TpetraMatrix)");

  dolfin_assert(!_matA.is_null());
  if (mode == "add" or mode == "insert" or mode == "flush")
    _matA->fillComplete();
  else
  {
    dolfin_error("TpetraMatrix.cpp",
                 "apply changes to Tpetra matrix",
                 "Unknown apply mode \"%s\"", mode.c_str());
  }
}
//-----------------------------------------------------------------------------
MPI_Comm TpetraMatrix::mpi_comm() const
{
  dolfin_assert(!_matA.is_null());
  // Unwrap MPI_Comm
  const Teuchos::RCP<const Teuchos::MpiComm<int>> _mpi_comm
    = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(_matA->getComm());

  return *(_mpi_comm->getRawMpiComm());
}
//-----------------------------------------------------------------------------
void TpetraMatrix::zero()
{
  dolfin_assert(!_matA.is_null());
  if(_matA->isFillComplete())
    _matA->resumeFill();

  _matA->setAllToScalar(0.0);
}
//-----------------------------------------------------------------------------
const TpetraMatrix& TpetraMatrix::operator*= (double a)
{
  dolfin_assert(!_matA.is_null());
  if(_matA->isFillComplete())
    _matA->resumeFill();

  _matA->scale(a);
  return *this;
}
//-----------------------------------------------------------------------------
const TpetraMatrix& TpetraMatrix::operator/= (double a)
{
  dolfin_assert(!_matA.is_null());
  if(_matA->isFillComplete())
    _matA->resumeFill();

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
    s << "< " << _matA->description() << " >";
  else
    s << "<TpetraMatrix of size " << size(0) << " x " << size(1) << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void TpetraMatrix::graphdump(const Teuchos::RCP<const graph_type> graph)
{
  int mpi_rank = graph->getRowMap()->getComm()->getRank();

  const Teuchos::RCP<const map_type> g_row_map = graph->getRowMap();
  const Teuchos::RCP<const map_type> g_col_map = graph->getColMap();
  const Teuchos::RCP<const map_type> domain_map = graph->getDomainMap();
  const Teuchos::RCP<const map_type> range_map = graph->getRangeMap();

  int n = g_row_map->getMaxAllGlobalIndex() + 1;
  int m = g_col_map->getMaxAllGlobalIndex() + 1;

  std::stringstream ss;
  ss << "RANK: " << mpi_rank << "\n";

  ss << "\n    ";
  for (int j = 0; j != m ; ++j)
  {
    if (g_col_map->isNodeGlobalElement(j))
      ss << "X";
    else
      ss << " ";
  }
  ss << "\n    ";
  for (int j = 0; j != m ; ++j)
  {
    if (domain_map->isNodeGlobalElement(j))
      ss << "O";
    else
      ss << " ";
  }
  ss << "\n----";
  for (int j = 0; j != m ; ++j)
    ss << "-";
  ss << "\n";

  for (int k = 0; k != n; ++k)
  {
    ss << " ";
    if (g_row_map->isNodeGlobalElement(k))
      ss << "X";
    else
      ss << " ";
    if (range_map->isNodeGlobalElement(k))
      ss << "O";
    else
      ss << " ";

    ss << "|";

    std::vector<bool> output_row(m, false);
    std::size_t nrow = graph->getNumEntriesInGlobalRow(k);
    if (nrow != Teuchos::OrdinalTraits<std::size_t>::invalid())
    {
      std::size_t nelem;
      std::vector<dolfin::la_index> row(nrow);
      Teuchos::ArrayView<dolfin::la_index> _row(row);
      graph->getGlobalRowCopy(k, _row, nelem);

      for (std::size_t j = 0; j != nrow; ++j)
        output_row[_row[j]] = true;
    }

    for (int j = 0; j != m; ++j)
    {
      if (output_row[j])
        ss << "x";
      else
        ss << " ";
    }
    ss << "\n";
  }

  std::cout << ss.str();
}
//-----------------------------------------------------------------------------

#endif
