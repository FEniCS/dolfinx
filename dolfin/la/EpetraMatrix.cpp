// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring
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
// Modified by Anders Logg 2008-2011
// Modified by Garth N. Wells 2008-2010
//
// First added:  2008-04-21
// Last changed: 2011-10-03

#ifdef HAS_TRILINOS

#include <cstring>
#include <iostream>
#include <iomanip>
#include <set>
#include <sstream>
#include <utility>

#include <Epetra_BlockMap.h>
#include <Epetra_CrsGraph.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_FECrsGraph.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_FEVector.h>
#include <Epetra_MpiComm.h>
#include <Epetra_SerialComm.h>
#include <EpetraExt_MatrixMatrix.h>

#include <dolfin/common/MPI.h>
#include <dolfin/log/dolfin_log.h>
#include "EpetraVector.h"
#include "GenericSparsityPattern.h"
#include "SparsityPattern.h"
#include "EpetraSparsityPattern.h"
#include "EpetraFactory.h"
#include "EpetraMatrix.h"


using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(const EpetraMatrix& A)
{
  if (A.mat())
    this->A.reset(new Epetra_FECrsMatrix(*A.mat()));
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(boost::shared_ptr<Epetra_FECrsMatrix> A) : A(A)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraMatrix::EpetraMatrix(const Epetra_CrsGraph& graph)
    : A(new Epetra_FECrsMatrix(Copy, graph))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraMatrix::~EpetraMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool EpetraMatrix::distributed() const
{
  assert(A);
  return A->Graph().DistributedGlobal();
}
//-----------------------------------------------------------------------------
void EpetraMatrix::init(const GenericSparsityPattern& sparsity_pattern)
{
  if (A && !A.unique())
    error("Cannot initialise EpetraMatrix. More than one object points to the underlying Epetra object.");

  // Get local range
  const std::pair<uint, uint> range = sparsity_pattern.local_range(0);
  const uint num_local_rows = range.second - range.first;
  const uint n0 = range.first;

  const SparsityPattern& _pattern = dynamic_cast<const SparsityPattern&>(sparsity_pattern);
  const std::vector<std::vector<dolfin::uint> > d_pattern = _pattern.diagonal_pattern(SparsityPattern::unsorted);
  const std::vector<std::vector<dolfin::uint> > o_pattern = _pattern.off_diagonal_pattern(SparsityPattern::unsorted);

  // Get number of non-zeroes per row (on and off diagonal)
  std::vector<uint> dnum_nonzeros, onum_nonzeros;
  sparsity_pattern.num_nonzeros_diagonal(dnum_nonzeros);
  sparsity_pattern.num_nonzeros_off_diagonal(onum_nonzeros);

  // Create row map
  EpetraFactory& f = EpetraFactory::instance();
  Epetra_MpiComm comm = f.get_mpi_comm();
  Epetra_Map row_map(sparsity_pattern.size(0), num_local_rows, 0, comm);

  // For rectangular matrices with more columns than rows, the columns which are
  // larger than those in row_map are marked as nonlocal (and assembly fails).
  // The domain_map fixes that problem, at least in the serial case.
  // FIXME: Needs attention in the parallel case. Maybe range_map is also req'd.
  const std::pair<uint, uint> colrange = sparsity_pattern.local_range(1);
  const int num_local_cols = colrange.second - colrange.first;
  Epetra_Map domain_map(sparsity_pattern.size(1), num_local_cols, 0, comm);

  // Create Epetra_FECrsGraph
  Epetra_FECrsGraph matrix_map(Copy, row_map, reinterpret_cast<int*>(&dnum_nonzeros[0]));

  // Add diagonal block indices
  std::vector<std::vector<dolfin::uint> >::const_iterator row_set;
  for (row_set = d_pattern.begin(); row_set != d_pattern.end(); ++row_set)
  {
    const uint global_row = row_set - d_pattern.begin() + n0;
    const std::vector<dolfin::uint>& nz_entries = *row_set;
    std::vector<dolfin::uint>& _nz_entries = const_cast<std::vector<dolfin::uint>& >(nz_entries);
    matrix_map.InsertGlobalIndices(global_row, row_set->size(),
                                   reinterpret_cast<int*>(&_nz_entries[0]));
  }

  for (uint local_row = 0; local_row < d_pattern.size(); local_row++)
  {
    const uint global_row = local_row + n0;
    std::vector<uint> &entries = const_cast<std::vector<uint>&>(d_pattern[local_row]);
    matrix_map.InsertGlobalIndices(global_row, entries.size(),
                                   reinterpret_cast<int*>(&entries[0]));
  }

  // Add off-diagonal block indices (parallel only)
  for (uint local_row = 0; local_row < o_pattern.size(); local_row++)
  {
    const uint global_row = local_row + n0;
    std::vector<uint> &entries = const_cast<std::vector<uint>&>(o_pattern[local_row]);
    matrix_map.InsertGlobalIndices(global_row, entries.size(),
                                   reinterpret_cast<int*>(&entries[0]));
  }

  try
  {
    // Finalise map. Here, row_map is standing in for RangeMap, which is
    // probably ok but should be double-checked.
    matrix_map.GlobalAssemble(domain_map, row_map);
    matrix_map.OptimizeStorage();
  }
  catch (int err)
  {
    error("Epetra threw error %d in assemble", err);
  }

  // Create matrix
  A.reset(new Epetra_FECrsMatrix(Copy, matrix_map));
}
//-----------------------------------------------------------------------------
EpetraMatrix* EpetraMatrix::copy() const
{
  return new EpetraMatrix(*this);
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraMatrix::size(uint dim) const
{
  assert(dim < 2);
  if (A)
  {
    const int M = A->NumGlobalRows();
    const int N = A->NumGlobalCols();
    return (dim == 0 ? M : N);
  }
  else
    return 0;
}
//-----------------------------------------------------------------------------
std::pair<dolfin::uint, dolfin::uint> EpetraMatrix::local_range(uint dim) const
{
  assert(dim < 2);
  if (dim == 1)
    error("Cannot compute columns range for Epetra matrices.");

  const Epetra_BlockMap& row_map = A->RowMap();
  assert(row_map.LinearMap());

  return std::make_pair(row_map.MinMyGID(), row_map.MaxMyGID() + 1);
}
//-----------------------------------------------------------------------------
void EpetraMatrix::resize(GenericVector& y, uint dim) const
{
  assert(A);

  // Get map appropriate map
  const Epetra_Map* map = 0;
  if (dim == 0)
    map = &(A->RangeMap());
  else if (dim == 1)
    map = &(A->DomainMap());
  else
    error("dim must be <= 1 to resize vector.");

  // Reset vector with new map
  EpetraVector& _y = y.down_cast<EpetraVector>();
  _y.reset(*map);
}
//-----------------------------------------------------------------------------
void EpetraMatrix::get(double* block, uint m, const uint* rows,
		                   uint n, const uint* cols) const
{
  assert(A);

  int num_entities = 0;
  int* indices;
  double* values;

  // For each row in rows
  for(uint i = 0; i < m; ++i)
  {
    // Extract the values and indices from row: rows[i]
    if (A->IndicesAreLocal())
    {
      const int err = A->ExtractMyRowView(rows[i], num_entities, values,
                                          indices);
      if (err != 0)
        error("EpetraMatrix::get: Did not manage to perform Epetra_CrsMatrix::ExtractMyRowView.");
    }
    else
    {
      const int err = A->ExtractGlobalRowView(rows[i], num_entities, values, indices);
      if (err != 0)
        error("EpetraMatrix::get: Did not manage to perform Epetra_CRSMatrix::ExtractGlobalRowView.");
    }

    // Step the indices to the start of cols
    int k = 0;
    while (indices[k] < static_cast<int>(cols[0]))
      k++;

    // Fill the collumns in the block
    for (uint j = 0; j < n; j++)
    {
      if (k < num_entities and indices[k] == static_cast<int>(cols[j]))
      {
        block[i*n + j] = values[k];
        k++;
      }
      else
        block[i*n + j] = 0.0;
    }
  }
}
//-----------------------------------------------------------------------------
void EpetraMatrix::set(const double* block,
                       uint m, const uint* rows,
                       uint n, const uint* cols)
{
  assert(A);

  // Work around for a bug in Trilinos 10.8 (see Bug lp 864510)
  /*
  for (uint i = 0; i < m; ++i)
  {
    const uint row = rows[i];
    const double* values = block + i*n;
    const int err = A->ReplaceGlobalValues(row, n, values,
                                           reinterpret_cast<const int*>(cols));
    assert(!err);
  }
  */

  const int err = A->ReplaceGlobalValues(m, reinterpret_cast<const int*>(rows),
                                   n, reinterpret_cast<const int*>(cols), block,
                                   Epetra_FECrsMatrix::ROW_MAJOR);
  if (err != 0)
    error("EpetraMatrix::set: Did not manage to perform Epetra_CrsMatrix::ReplaceGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::add(const double* block,
                       uint m, const uint* rows,
                       uint n, const uint* cols)
{
  assert(A);

  // Work around for a bug in Trilinos 10.8 (see Bug lp 864510)
  /*
  for (uint i = 0; i < m; ++i)
  {
    const uint row = rows[i];
    const double* values = block + i*n;
    const int err = A->SumIntoGlobalValues(row, n, values,
                                           reinterpret_cast<const int*>(cols));
    assert(!err);
  }
  */

  const int err = A->SumIntoGlobalValues(m, reinterpret_cast<const int*>(rows),
                                         n, reinterpret_cast<const int*>(cols), block,
                                         Epetra_FECrsMatrix::ROW_MAJOR);
  if (err != 0)
    error("EpetraMatrix::add: Did not manage to perform Epetra_CrsMatrix::SumIntoGlobalValues.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::axpy(double a, const GenericMatrix& A, bool same_nonzero_pattern)
{
  const EpetraMatrix* AA = &A.down_cast<EpetraMatrix>();
  if (!AA->mat()->Filled())
    error("Epetramatrix is not in the correct state for addition.");

  const int err = EpetraExt::MatrixMatrix::Add(*(AA->mat()), false, a, *(this->A), 1.0);
  if (err != 0)
    error("EpetraMatrDid::axpy: Did not manage to perform EpetraExt::MatrixMatrix::Add. If the matrix has been assembled, the nonzero patterns must match.");
}
//-----------------------------------------------------------------------------
double EpetraMatrix::norm(std::string norm_type) const
{
  if (norm_type == "l1")
    return A->NormOne();
  else if (norm_type == "linf")
    return A->NormInf();
  else if (norm_type == "frobenius")
    return A->NormFrobenius();
  else
  {
    error("Unknown norm type in EpetraMatrix.");
    return 0.0;
  }
}
//-----------------------------------------------------------------------------
void EpetraMatrix::zero()
{
  assert(A);
  const int err = A->PutScalar(0.0);
  if (err != 0)
    error("EpetraMatrix::zero: Did not manage to perform Epetra_CrsMatrix::PutScalar.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::apply(std::string mode)
{
  assert(A);
  int err = 0;
  if (mode == "add")
    err = A->GlobalAssemble(Add);
  else if (mode == "insert")
    err = A->GlobalAssemble(Insert);
  else
  {
    dolfin_error("EpetraMatrix.cpp",
                 "apply changes to matrix",
                 "Unknown apply mode \"%s\"", mode.c_str());
  }

  if (err != 0)
    error("EpetraMatrix::apply: Did not manage to perform Epetra_CrsMatrix::GlobalAssemble.");
}
//-----------------------------------------------------------------------------
std::string EpetraMatrix::str(bool verbose) const
{
  assert(A);
  std::stringstream s;
  if (verbose)
  {
    warning("Verbose output for EpetraMatrix not implemented, calling Epetra Print directly.");
    A->Print(std::cout);
  }
  else
    s << "<EpetraMatrix of size " << size(0) << " x " << size(1) << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void EpetraMatrix::ident(uint m, const uint* rows)
{
  assert(A);
  assert(A->Filled() == true);

  // FIXME: This is a major hack and will not scale for large numbers of
  // processes. The problem is that a dof is not guaranteed to reside on
  // the same process as one of cells to which it belongs (which is bad,
  // but is due to the sparsity pattern computation). This function only
  // work for locally owned rows (the PETSc version works for any row).

  typedef boost::unordered_set<uint> MySet;

  // Build lists of local and nonlocal rows
  MySet local_rows;
  std::vector<uint> non_local_rows;
  for (uint i = 0; i < m; ++i)
  {
    if (A->MyGlobalRow(rows[i]))
      local_rows.insert(rows[i]);
    else
      non_local_rows.push_back(rows[i]);
  }

  // If parallel, send non_local rows to all processes
  if (MPI::num_processes() > 1)
  {
    // Send list of nonlocal rows to all processes
    std::vector<uint> destinations;
    std::vector<uint> send_data;
    for (uint i = 0; i < MPI::num_processes(); ++i)
    {
      if (i != MPI::process_number())
      {
        send_data.insert(send_data.end(), non_local_rows.begin(),
                             non_local_rows.end());
        destinations.insert(destinations.end(), non_local_rows.size(), i);
      }
    }

    std::vector<uint> received_data;
    MPI::distribute(send_data, destinations, received_data);

    // Unpack data
    for (uint i = 0; i < received_data.size(); ++i)
    {
      // Insert row into set if it's local
      const uint new_index = received_data[i];
      if (A->MyGlobalRow(new_index))
        local_rows.insert(new_index);
    }
  }
  //-------------------------

  const Epetra_CrsGraph& graph = A->Graph();
  MySet::const_iterator global_row;
  for (global_row = local_rows.begin(); global_row != local_rows.end(); ++global_row)
  {
    // Get local row index
    const int local_row = A->LRID(*global_row);

    // If this process owns row, then zero row
    if (local_row >= 0)
    {
      // Get row map
      int num_nz = 0;
      int* column_indices;
      int err = graph.ExtractMyRowView(local_row, num_nz, column_indices);
      if (err != 0)
        error("EpetraMatrix::ident: Did not manage to extract row map.");

      // Zero row
      std::vector<double> block(num_nz, 0.0);
      err = A->ReplaceMyValues(local_row, num_nz, &block[0], column_indices);
      if (err != 0)
        error("EpetraMatrix::ident: Did not manage to perform Epetra_CrsMatrix::ReplaceGlobalValues.");

      // Place one on the diagonal
      const double one = 1.0;
      A->ReplaceMyValues(local_row, 1, &one, &local_row);
    }
  }
}
//-----------------------------------------------------------------------------
void EpetraMatrix::zero(uint m, const uint* rows)
{
  // FIXME: This can be made more efficient by eliminating creation of some
  //        obejcts inside the loop

  assert(A);
  const Epetra_CrsGraph& graph = A->Graph();
  for (uint i = 0; i < m; ++i)
  {
    const int row = rows[i];
    const int num_nz = graph.NumGlobalIndices(row);
    std::vector<int> indices(num_nz);

    int out_num = 0;
    graph.ExtractGlobalRowCopy(row, num_nz, out_num, &indices[0]);

    std::vector<double> block(num_nz);
    const int err = A->ReplaceGlobalValues(row, num_nz, &block[0], &indices[0]);
    if (err != 0)
      error("EpetraMatrix::zero: Did not manage to perform Epetra_CrsMatrix::ReplaceGlobalValues.");
  }
}
//-----------------------------------------------------------------------------
void EpetraMatrix::mult(const GenericVector& x_, GenericVector& Ax_) const
{
  assert(A);
  const EpetraVector& x = x_.down_cast<EpetraVector>();
  EpetraVector& Ax = Ax_.down_cast<EpetraVector>();

  if (x.size() != size(1))
    error("EpetraMatrix::mult: Matrix and vector dimensions don't match for matrix-vector product.");

  // Resize RHS
  this->resize(Ax, 0);

  assert(x.vec());
  assert(Ax.vec());
  const int err = A->Multiply(false, *(x.vec()), *(Ax.vec()));
  if (err != 0)
    error("EpetraMatrix::mult: Did not manage to perform Epetra_CRSMatrix::Multiply.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::transpmult(const GenericVector& x_, GenericVector& Ax_) const
{
  assert(A);
  const EpetraVector& x = x_.down_cast<EpetraVector>();
  EpetraVector& Ax = Ax_.down_cast<EpetraVector>();

  if (x.size() != size(0))
    error("EpetraMatrix::transpmult: Matrix and vector dimensions don't match for (transposed) matrix-vector product.");

  // Resize RHS
  this->resize(Ax, 1);

  const int err = A->Multiply(true, *(x.vec()), *(Ax.vec()));
  if (err != 0)
    error("EpetraMatrix::transpmult: Did not manage to perform Epetra_CRSMatrix::Multiply.");
}
//-----------------------------------------------------------------------------
void EpetraMatrix::getrow(uint row, std::vector<uint>& columns,
                          std::vector<double>& values) const
{
  assert(A);

  // Get local row index
  const int local_row_index = A->LRID(row);

  // If this process has part of the row, get values
  if (local_row_index >= 0)
  {
    // Temporary variables
    int* indices;
    double* vals;
    int num_entries;

    // Extract data from Epetra matrix
    const int err = A->ExtractMyRowView(local_row_index, num_entries, vals, indices);
    if (err != 0)
      error("EpetraMatrix::getrow: Did not manage to perform Epetra_CrsMatrix::ExtractMyRowView.");

    // Put data in columns and values
    columns.resize(num_entries);
    values.resize(num_entries);
    for (int i = 0; i < num_entries; i++)
    {
      columns[i] = indices[i];
      values[i]  = vals[i];
    }
  }
  else
  {
    columns.resize(0);
    values.resize(0);
  }
}
//-----------------------------------------------------------------------------
void EpetraMatrix::setrow(uint row, const std::vector<uint>& columns,
                          const std::vector<double>& values)
{
  static bool print_msg_once=true;
  if (print_msg_once)
  {
    info("EpetraMatrix::setrow is implemented inefficiently");
    print_msg_once = false;
  }

  for (uint i=0; i < columns.size(); i++)
    set(&values[i], 1, &row, 1, &columns[i]);
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& EpetraMatrix::factory() const
{
  return EpetraFactory::instance();
}
//-----------------------------------------------------------------------------
void EpetraMatrix::init(const EpetraSparsityPattern& sparsity_pattern)
{
  if (A && !A.unique())
    error("Cannot initialise EpetraMatrix. More than one object points to the underlying Epetra object.");
  A.reset(new Epetra_FECrsMatrix(Copy, sparsity_pattern.pattern()));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Epetra_FECrsMatrix> EpetraMatrix::mat() const
{
  return A;
}
//-----------------------------------------------------------------------------
const EpetraMatrix& EpetraMatrix::operator*= (double a)
{
  assert(A);
  const int err = A->Scale(a);
  if (err !=0)
    error("EpetraMatrix::operator*=: Did not manage to perform Epetra_CrsMatrix::Scale.");
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraMatrix& EpetraMatrix::operator/= (double a)
{
  assert(A);
  int err = A->Scale(1.0/a);
  if (err !=0)
    error("EpetraMatrix::operator/=: Did not manage to perform Epetra_CrsMatrix::Scale.");
  return *this;
}
//-----------------------------------------------------------------------------
const GenericMatrix& EpetraMatrix::operator= (const GenericMatrix& A)
{
  *this = A.down_cast<EpetraMatrix>();
  return *this;
}
//-----------------------------------------------------------------------------
const EpetraMatrix& EpetraMatrix::operator= (const EpetraMatrix& A)
{
  if (A.mat())
    this->A.reset(new Epetra_FECrsMatrix(*A.mat()));
  else
    A.mat().reset();

  return *this;
}
//-----------------------------------------------------------------------------
#endif
