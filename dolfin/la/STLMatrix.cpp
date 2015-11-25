// Copyright (C) 2007-2012 Anders Logg and Garth N. Wells
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
// Modified by Ola Skavhaug 2007
// Modified by Ilmar Wilbers 2008
//
// First added:  2007-01-17
// Last changed: 2012-03-15

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <boost/serialization/utility.hpp>

#include <dolfin/common/Timer.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include "STLFactory.h"
#include "STLFactoryCSC.h"
#include "STLMatrix.h"

using namespace dolfin;

struct CompareIndex
{
  CompareIndex(std::size_t index) : _index(index) {}
  bool operator()(const std::pair<std::size_t, double>& entry) const
  { return _index == entry.first; }
  private:
    const std::size_t _index;
};

//-----------------------------------------------------------------------------
void STLMatrix::init(const TensorLayout& tensor_layout)
{
  // Check that sparsity pattern has correct storage (row vs column storage)
  if (_primary_dim != tensor_layout.primary_dim)
  {
    dolfin_error("STLMatrix.cpp",
                 "initialization of STL matrix",
                 "Primary storage dim of matrix and tensor layout must be the same");
  }

  // Get MPI communicator
  _mpi_comm = tensor_layout.mpi_comm();

  // Set co-dimension
  std::size_t primary_codim = 1;
  if (_primary_dim == 1)
    primary_codim = 0;

  // Set block size
  _block_size = tensor_layout.index_map(0)->block_size();
  dolfin_assert(_block_size == tensor_layout.index_map(1)->block_size());

  _local_range = tensor_layout.local_range(_primary_dim);
  num_codim_entities = tensor_layout.size(primary_codim);

  const std::size_t num_primary_entiries = _local_range.second
    - _local_range.first;

  _values.resize(num_primary_entiries);

  // FIXME: Add function to sparsity pattern to get nnz per row to
  //        to reserve space for vectors
  //if (tensor_layout.sparsity_pattern()
  //{
  //  Reserve space here
  //}
}
//-----------------------------------------------------------------------------
std::size_t STLMatrix::size(std::size_t dim) const
{
  if (dim > 1)
  {
    dolfin_error("STLMatrix.cpp",
                 "access size of STL matrix",
                 "Illegal axis (%d), must be 0 or 1", dim);
  }

  if (_primary_dim == 0)
  {
    if (dim == 0)
    {
      return dolfin::MPI::sum(_mpi_comm,
                              _local_range.second - _local_range.first);
    }
    else
      return num_codim_entities;
  }
  else
  {
    if (dim == 0)
      return num_codim_entities;
    else
    {
      return dolfin::MPI::sum(_mpi_comm,
                              _local_range.second - _local_range.first);
    }
  }
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, std::size_t>
STLMatrix::local_range(std::size_t dim) const
{
  dolfin_assert(dim < 2);
  if (_primary_dim == 0)
  {
    if (dim == 0)
      return _local_range;
    else
      return std::make_pair(0, num_codim_entities);
  }
  else
  {
    if (dim == 0)
      return std::make_pair(0, num_codim_entities);
    else
      return _local_range;
  }
}
//-----------------------------------------------------------------------------
void STLMatrix::zero()
{
  std::vector<std::vector<std::pair<std::size_t, double>>>::iterator slice;
  std::vector<std::pair<std::size_t, double>>::iterator entry;
  for (slice = _values.begin(); slice != _values.end(); ++slice)
    for (entry = slice->begin(); entry != slice->end(); ++entry)
      entry->second = 0.0;
}
//-----------------------------------------------------------------------------
void STLMatrix::add(const double* block,
                    std::size_t m, const dolfin::la_index* rows,
                    std::size_t n, const dolfin::la_index* cols)
{
  // Perform a simple linear search along each column. Otherwise,
  // append the value (calling push_back).

  const dolfin::la_index* primary_slice = rows;
  const dolfin::la_index* secondary_slice = cols;

  std::size_t dim   = m;
  std::size_t codim = n;
  std::size_t map0  = 1;
  std::size_t map1  = n;
  if (_primary_dim == 1)
  {
    dim = n;
    codim = m;
    map0  = n;
    map1  = 1;
  }

  // Iterate over primary dimension
  for (std::size_t i = 0; i < dim; i++)
  {
    // Global primary index
    const std::size_t I = primary_slice[i];

    // Check if I is a local row/column
    if (I < _local_range.second && I >= _local_range.first)
    {
      const std::size_t I_local = I - _local_range.first;

      assert(I_local < _values.size());
      std::vector<std::pair<std::size_t, double>>& slice = _values[I_local];

      // Iterate over co-dimension
      for (std::size_t j = 0; j < codim; j++)
      {
        const std::size_t pos = i*map1 + j*map0;

        // Global index
        const std::size_t J = secondary_slice[j];

        // Check if entry exists and insert
        std::vector<std::pair<std::size_t, double>>::iterator entry
              = std::find_if(slice.begin(), slice.end(), CompareIndex(J));
        if (entry != slice.end())
          entry->second += block[pos];
        else
          slice.push_back(std::make_pair(J, block[pos]));
      }
    }
    else
    {
      // Iterate over columns
      for (std::size_t j = 0; j < n; j++)
      {
        // Global column, coordinate
        const std::size_t J = secondary_slice[j];
        const std::pair<std::size_t, std::size_t> global_coordinate(I, J);
        //const std::size_t pos = i*n + j;
        const std::size_t pos = i*map1 + j*map0;

        boost::unordered_map<std::pair<std::size_t,
                                       std::size_t>, double>::iterator coord;
        coord = off_processs_data.find(global_coordinate);
        if (coord == off_processs_data.end())
          off_processs_data[global_coordinate] = block[pos];
        else
          coord->second += block[pos];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void STLMatrix::apply(std::string mode)
{
  Timer timer("Apply (STLMatrix)");

  // Number of processes
  const std::size_t num_processes = MPI::size(_mpi_comm);

  // Data to send
  std::vector<std::vector<std::size_t>> send_non_local_rows(num_processes);
  std::vector<std::vector<std::size_t>> send_non_local_cols(num_processes);
  std::vector<std::vector<double>> send_non_local_vals(num_processes);

  std::vector<std::size_t> range(2);
  range[0] = _local_range.first;
  range[1] = _local_range.second;
  std::vector<std::size_t> process_ranges;
  dolfin::MPI::all_gather(_mpi_comm, range, process_ranges);

  // Communicate off-process data
  boost::unordered_map<std::pair<std::size_t,
                               std::size_t>, double>::const_iterator entry;
  for (entry = off_processs_data.begin(); entry != off_processs_data.end();
       ++entry)
  {
    const std::size_t global_row = entry->first.first;

    // FIXME: This can be more efficient by storing sparsity pattern,
    //        or caching owning process for repeated assembly

    // Get owning process
    std::size_t owner = 0;
    for (std::size_t proc = 0; proc < num_processes; ++proc)
    {
      if (global_row < process_ranges[2*proc + 1]
          && global_row >= process_ranges[2*proc])
      {
        owner = proc;
        break;
      }
    }

    send_non_local_rows[owner].push_back(global_row);
    send_non_local_cols[owner].push_back(entry->first.second);
    send_non_local_vals[owner].push_back(entry->second);
  }

  // Send/receive data
  std::vector<std::vector<std::size_t>> received_non_local_rows;
  std::vector<std::vector<std::size_t>> received_non_local_cols;
  std::vector<std::vector<double>> received_non_local_vals;
  dolfin::MPI::all_to_all(_mpi_comm, send_non_local_rows,
                          received_non_local_rows);
  dolfin::MPI::all_to_all(_mpi_comm, send_non_local_cols,
                          received_non_local_cols);
  dolfin::MPI::all_to_all(_mpi_comm, send_non_local_vals,
                          received_non_local_vals);

  // Add/insert off-process data
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    const std::vector<std::size_t>& received_non_local_rows_p
      = received_non_local_rows[p];
    const std::vector<std::size_t>& received_non_local_cols_p
      = received_non_local_cols[p];
    const std::vector<double>& received_non_local_vals_p
      = received_non_local_vals[p];

    dolfin_assert(received_non_local_rows_p.size()
           == received_non_local_cols_p.size());
    dolfin_assert(received_non_local_rows_p.size()
           == received_non_local_vals_p.size());

    for (std::size_t i = 0; i < received_non_local_rows_p.size(); ++i)
    {
      dolfin_assert(received_non_local_rows_p[i] < _local_range.second
                    && received_non_local_rows_p[i] >= _local_range.first);
      const std::size_t I_local
        = received_non_local_rows_p[i] - _local_range.first;
      dolfin_assert(I_local < _values.size());

      const std::size_t J = received_non_local_cols_p[i];
      std::vector<std::pair<std::size_t, double>>::iterator e
            = std::find_if(_values[I_local].begin(), _values[I_local].end(),
                           CompareIndex(J));
      if (e != _values[I_local].end())
        e->second += received_non_local_vals_p[i];
      else
        _values[I_local].push_back(std::make_pair(J,
                                               received_non_local_vals_p[i]));
    }
  }

  // Sort columns (csr)/rows (csc)
  sort();
}
//-----------------------------------------------------------------------------
double STLMatrix::norm(std::string norm_type) const
{
  if (norm_type != "frobenius")
    dolfin_error("STLMatrix.cpp",
                 "compute matrix norm",
                 "Do not know to compute %s norm for STLMatrix",
                 norm_type.c_str());

  double _norm = 0.0;
  for (std::size_t i = 0; i < _values.size(); ++i)
  {
    for (std::size_t j = 0; j < _values[i].size(); ++j)
      _norm += _values[i][j].second*_values[i][j].second;
  }
  return std::sqrt(dolfin::MPI::sum(_mpi_comm, _norm));
}
//-----------------------------------------------------------------------------
void STLMatrix::getrow(std::size_t row, std::vector<std::size_t>& columns,
                       std::vector<double>& values) const
{
  if (_primary_dim == 1)
  {
    dolfin_error("STLMatrix.cpp",
                 "getting row from matrix",
                 "A row can only be extract from a STLMatrix that use row-wise storage.");
  }

  dolfin_assert(row < _local_range.second && row >= _local_range.first);
  const std::size_t local_row = row - _local_range.first;
  dolfin_assert(local_row < _values.size());

  // Copy row values
  columns.resize(_values[local_row].size());
  values.resize(_values[local_row].size());
  for (std::size_t i = 0; i < _values[local_row].size(); ++i)
  {
    columns[i] = _values[local_row][i].first;
    values[i]  = _values[local_row][i].second;
  }
}
//-----------------------------------------------------------------------------
void STLMatrix::ident(std::size_t m, const dolfin::la_index* rows)
{
  if (_primary_dim == 1)
  {
    dolfin_error("STLMatrix.cpp",
                 "creating identity row",
                 "STLMatrix::ident can only be used with row-wise storage.");
  }

  std::pair<std::size_t, std::size_t> row_range = local_range(0);
  for (std::size_t i = 0; i < m; ++i)
  {
    const std::size_t global_row = rows[i];
    if (global_row >= row_range.first && global_row < row_range.second)
    {
      const std::size_t local_row = global_row - row_range.first;
      dolfin_assert(local_row < _values.size());
      for (std::size_t j = 0; j < _values[local_row].size(); ++j)
        _values[local_row][j].second = 0.0;

      // Place one on diagonal
      std::vector<std::pair<std::size_t, double>>::iterator diagonal
          = std::find_if(_values[local_row].begin(), _values[local_row].end(),
                         CompareIndex(global_row));

      if (diagonal != _values[local_row].end())
        diagonal->second = 1.0;
      else
        _values[local_row].push_back(std::make_pair(global_row, 1.0));
    }
  }
}
//-----------------------------------------------------------------------------
const STLMatrix& STLMatrix::operator*= (double a)
{
  std::vector<std::vector<std::pair<std::size_t, double>>>::iterator row;
  std::vector<std::pair<std::size_t, double>>::iterator entry;
  for (row = _values.begin(); row != _values.end(); ++row)
    for (entry = row->begin(); entry != row->end(); ++entry)
      entry->second *=a;

  return *this;
}
//-----------------------------------------------------------------------------
const STLMatrix& STLMatrix::operator/= (double a)
{
  return (*this) *= 1.0/a;
}
//-----------------------------------------------------------------------------
std::string STLMatrix::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    if (_primary_dim == 1)
    {
      dolfin_error("STLMatrix.cpp",
                   "verbose string output of matrix",
                   "Verbose string output is currently supported for row-wise storage only.");
    }

    s << str(false) << std::endl << std::endl;
    for (std::size_t i = 0; i < _local_range.second - _local_range.first; i++)
    {
      // Sort row data by column index
      std::vector<std::pair<std::size_t, double>> data = _values[i];
      std::sort(data.begin(), data.end());

      // Set precision
      std::stringstream line;
      line << std::setiosflags(std::ios::scientific);
      line << std::setprecision(1);

      // Format matrix
      line << "|";
      std::vector<std::pair<std::size_t, double>>::const_iterator entry;
      for (entry = data.begin(); entry != data.end(); ++entry)
      {
        line << " (" << i << ", " << entry->first << ", " << entry->second
             << ")";
      }
      line << " |";

      s << line.str().c_str() << std::endl;
    }
  }
  else
    s << "<STLMatrix of size " << size(0) << " x " << size(1) << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& STLMatrix::factory() const
{
  if (_primary_dim == 0)
    return STLFactory::instance();
  else
    return STLFactoryCSC::instance();
}
//-----------------------------------------------------------------------------
std::size_t STLMatrix::nnz() const
{
  return dolfin::MPI::sum(_mpi_comm, local_nnz());
}
//-----------------------------------------------------------------------------
std::size_t STLMatrix::local_nnz() const
{
  std::size_t _nnz = 0;
  for (std::size_t i = 0; i < _values.size(); ++i)
    _nnz += _values[i].size();
  return _nnz;
}
//-----------------------------------------------------------------------------
