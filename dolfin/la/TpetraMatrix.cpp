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
  if (!A._matA.is_null())
  {
    // FIXME - just copy mat and maps
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
  const std::size_t m = row_range.second - row_range.first;

  // Get sparsity pattern
  dolfin_assert(tensor_layout.sparsity_pattern());
  std::shared_ptr<const GenericSparsityPattern> sparsity_pattern
    = tensor_layout.sparsity_pattern();

  // Initialize matrix
  // Insist on square Matrix for now
  dolfin_assert(M == N);

  // Set up MPI Comm
  Teuchos::RCP<const Teuchos::Comm<int>>
    _comm(new Teuchos::MpiComm<int>(sparsity_pattern->mpi_comm()));

  // Save the local row and column mapping, so we can use add_local
  // and set_local later with off-process entries
  std::vector<dolfin::la_index> global_indices0
    (tensor_layout.local_to_global_map[0].begin(),
     tensor_layout.local_to_global_map[0].end());
  Teuchos::ArrayView<dolfin::la_index> _global_indices0(global_indices0);
  _row_map = Teuchos::rcp
    (new map_type(Teuchos::OrdinalTraits<dolfin::la_index>::invalid(),
                  _global_indices0, 0, _comm));

  std::vector<dolfin::la_index> global_indices1
    (tensor_layout.local_to_global_map[1].begin(),
     tensor_layout.local_to_global_map[1].end());
  Teuchos::ArrayView<dolfin::la_index> _global_indices1(global_indices1);
  _col_map = Teuchos::rcp
    (new map_type(Teuchos::OrdinalTraits<dolfin::la_index>::invalid(),
                  _global_indices1, 0, _comm));

  // Make a Tpetra::CrsGraph of the sparsity_pattern
  typedef Tpetra::CrsGraph<> graph_type;
  std::vector<std::vector<std::size_t>> pattern_diag
    = sparsity_pattern->diagonal_pattern(GenericSparsityPattern::unsorted);
  std::vector<std::vector<std::size_t>> pattern_off
    = sparsity_pattern->off_diagonal_pattern(GenericSparsityPattern::unsorted);

  dolfin_assert(pattern_diag.size() == pattern_off.size());
  dolfin_assert(m == pattern_diag.size());

  // Get number of non-zeros per row to allocate storage
  std::vector<std::size_t> entries_per_row(m);
  sparsity_pattern->num_local_nonzeros(entries_per_row);
  Teuchos::ArrayRCP<std::size_t> _nnz(entries_per_row.data(), 0,
                                      entries_per_row.size(), false);

  // Create a non-overlapping "row" map for the graph
  // The column map will be auto-generated from the entries.
  Teuchos::ArrayView<dolfin::la_index>
    _global_indices_subset(global_indices0.data(), m);
  Teuchos::RCP<const map_type> graph_row_map
    (new map_type(Teuchos::OrdinalTraits<dolfin::la_index>::invalid(),
                  _global_indices_subset, 0, _comm));

  Teuchos::RCP<graph_type> crs_graph
    (new graph_type(graph_row_map, _nnz, Tpetra::StaticProfile));

  for (std::size_t i = 0; i != m; ++i)
  {
    std::vector<dolfin::la_index> indices(pattern_diag[i].begin(),
                                          pattern_diag[i].end());
    indices.insert(indices.end(), pattern_off[i].begin(),
                   pattern_off[i].end());

    Teuchos::ArrayView<dolfin::la_index> _indices(indices);
    crs_graph->insertGlobalIndices(tensor_layout.local_to_global_map[0][i],
                                _indices);
  }

  crs_graph->fillComplete();
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
    num_elements = _matA->getRangeMap()->getGlobalNumElements();
  else
    num_elements = _matA->getDomainMap()->getGlobalNumElements();

  return num_elements;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
TpetraMatrix::local_range(std::size_t dim) const
{
  if (dim == 0)
  {
    Teuchos::RCP<const map_type> a_row_map(_matA->getRowMap());
    return std::make_pair<std::size_t, std::size_t>
      (a_row_map->getMinGlobalIndex(), a_row_map->getMaxGlobalIndex() + 1);
  }
  else if (dim == 1)
  {
    // FIXME: this is not quite right - column map will have overlap
    Teuchos::RCP<const map_type> a_col_map(_matA->getColMap());
    return std::make_pair<std::size_t, std::size_t>
      (a_col_map->getMinGlobalIndex(), a_col_map->getMaxGlobalIndex() + 1);
  }

  return std::make_pair(0,0);
}
//-----------------------------------------------------------------------------
std::size_t TpetraMatrix::nnz() const
{
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
    _map =_matA->getRangeMap();
  else if (dim == 1)
    _map = _matA->getDomainMap();
  else
  {
    dolfin_error("TpetraMatrix.cpp",
                 "initialize Tpetra vector to match Tpetra matrix",
                 "Dimension must be 0 or 1, not %d", dim);
  }

  _z._x = Teuchos::rcp(new TpetraVector::vector_type(_map, 1));
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
    Teuchos::ArrayView<const dolfin::la_index> _columns;
    Teuchos::ArrayView<const double> _data;
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
    _global_col_idx.push_back(_col_map->getGlobalElement(cols[i]));
  Teuchos::ArrayView<const dolfin::la_index> global_col_idx(_global_col_idx);

  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const double> data(block + i*n, n);

    const dolfin::la_index global_row_idx
      = _row_map->getGlobalElement(rows[i]);
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
    _global_col_idx.push_back(_col_map->getGlobalElement(cols[i]));
  Teuchos::ArrayView<const dolfin::la_index> global_col_idx(_global_col_idx);

  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const double> data(block + i*n, n);

    const dolfin::la_index global_row_idx
      = _row_map->getGlobalElement(rows[i]);
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
  if (ncols == Tpetra::OrdinalTraits<std::size_t>::invalid())
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
    Teuchos::ArrayView<const dolfin::la_index> cols;
    Teuchos::ArrayView<const double> data;
    _matA->getGlobalRowView(rows[i], cols, data);
    std::vector<double> z(cols.size(), 0);
    Teuchos::ArrayView<const double> dataz(z);

    _matA->replaceGlobalValues(rows[i], cols, dataz);
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::zero_local(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(!_matA.is_null());
  dolfin_assert(!_matA->isFillComplete());

  for (std::size_t i = 0 ; i != m; ++i)
  {
    Teuchos::ArrayView<const dolfin::la_index> cols;
    Teuchos::ArrayView<const double> data;
    _matA->getLocalRowView(rows[i], cols, data);
    std::vector<double> z(cols.size(), 0);
    Teuchos::ArrayView<const double> dataz(z);

    _matA->replaceLocalValues(rows[i], cols, dataz);
  }
}
//-----------------------------------------------------------------------------
void TpetraMatrix::ident(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(!_matA.is_null());
  dolfin_assert(!_matA->isFillComplete());

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

  _matA->apply(*xx._x, *yy._x, Teuchos::TRANS);
}
//-----------------------------------------------------------------------------
void TpetraMatrix::set_diagonal(const GenericVector& x)
{
  dolfin_assert(!_matA.is_null());
  dolfin_assert(!_matA->isFillComplete());

  const TpetraVector& xx = x.down_cast<TpetraVector>();

  if (xx._x->getMap()->isSameAs(*_matA->getRowMap()))
    std::cout << "Should be OK\n";

  dolfin_not_implemented();

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
  {
    dolfin_error("TpetraMatrix.cpp",
                 "apply changes to Tpetra matrix",
                 "Unknown apply mode \"%s\"", mode.c_str());
  }
}
//-----------------------------------------------------------------------------
MPI_Comm TpetraMatrix::mpi_comm() const
{
  // Unwrap MPI_Comm
  const Teuchos::RCP<const Teuchos::MpiComm<int>> _mpi_comm
    = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(_matA->getComm());

  return *(_mpi_comm->getRawMpiComm());
}
//-----------------------------------------------------------------------------
void TpetraMatrix::zero()
{
  dolfin_assert(!_matA.is_null());
  dolfin_assert(!_matA->isFillComplete());
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
  dolfin_assert(!_matA->isFillComplete());
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
