// Copyright (C) 2007-2011 Anders Logg and Garth N. Wells
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
// Modified by Ola Skavhaug, 2007.
// Modified by Ilmar Wilbers, 2008.
//
// First added:  2007-01-17
// Last changed: 2011-10-29


#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include "STLFactory.h"
#include "STLMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void STLMatrix::init(const GenericSparsityPattern& sparsity_pattern)
{
  _local_range = sparsity_pattern.local_range(0);
  ncols = sparsity_pattern.size(1);

  const uint num_rows = _local_range.second - _local_range.first;
  _cols.resize(num_rows);
  _vals.resize(num_rows);

  // FIXME: Add function to sparsity pattern to get nnz per row to
  //        to reserve space for vectors
}
//-----------------------------------------------------------------------------
dolfin::uint STLMatrix::size(uint dim) const
{
  if (dim > 1)
  {
    dolfin_error("STLMatrix.cpp",
                 "access size of STL matrix",
                 "Illegal axis (%d), must be 0 or 1", dim);
  }

  if (dim == 0)
    return dolfin::MPI::sum(_local_range.second - _local_range.first);
  else
    return ncols;
}
//-----------------------------------------------------------------------------
std::pair<uint, uint> STLMatrix::local_range(uint dim) const
{
  dolfin_assert(dim < 2);
  if (dim == 0)
    return _local_range;
  else
    return std::make_pair(0, ncols);
}
//-----------------------------------------------------------------------------
void STLMatrix::zero()
{
  for (std::vector<std::vector<double> >::iterator row = _vals.begin(); row != _vals.end(); ++row)
    std::fill(row->begin(), row->end(), 0);
}
//-----------------------------------------------------------------------------
void STLMatrix::add(const double* block, uint m, const uint* rows, uint n,
                    const uint* cols)
{
  // Perform a simple linear search along each column. Otherwise,
  // append the value (calling push_back).

  // Iterate over rows
  for (uint i = 0; i < m; i++)
  {
    // Global row index
    const uint I = rows[i];

    // Check if I is a local row
    if (I < _local_range.second && I >= _local_range.first)
    {
      const uint I_local = I - _local_range.first;
      assert(I_local < this->_cols.size());
      assert(I_local < this->_vals.size());

      std::vector<uint>& rcols = this->_cols[I_local];
      std::vector<double>& rvals = this->_vals[I_local];

      // Iterate over columns
      for (uint j = 0; j < n; j++)
      {
        const uint pos = i*n + j;

        // Global column
        const uint J = cols[j];

        // Check if column entry exists and insert
        const std::vector<uint>::const_iterator column = std::find(rcols.begin(), rcols.end(), J);
        if (column != rcols.end())
          rvals[column - rcols.begin()] += block[pos];
        else
        {
          rcols.push_back(J);
          rvals.push_back(block[pos]);
        }
      }
    }
    else
    {
      // Iterate over columns
      for (uint j = 0; j < n; j++)
      {
        // Global column, coordinate
        const uint J = cols[j];
        const std::pair<uint, uint> global_coordinate(I, J);
        const uint pos = i*n + j;

        boost::unordered_map<std::pair<uint, uint>, double>::iterator coord;
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
  // Data to send
  std::vector<uint> send_non_local_rows, send_non_local_cols, destinations;
  std::vector<double> send_non_local_vals;

  std::vector<std::pair<uint, uint> > process_ranges;
  dolfin::MPI::all_gather(_local_range, process_ranges);

  // Communicate off-process data
  boost::unordered_map<std::pair<uint, uint>, double>::const_iterator entry;
  for (entry = off_processs_data.begin(); entry != off_processs_data.end(); ++entry)
  {
    const uint global_row = entry->first.first;

    // FIXME: This can be more efficient by storing sparsity pattern,
    //        or caching owning process for repeated assembly

    // Get owning process
    uint owner = 0;
    for (uint proc = 0; proc < process_ranges.size(); ++proc)
    {
      if (global_row < process_ranges[proc].second &&  global_row >= process_ranges[proc].first)
      {
        owner = proc;
        break;
      }
    }

    send_non_local_rows.push_back(global_row);
    send_non_local_cols.push_back(entry->first.second);
    send_non_local_vals.push_back(entry->second);
    destinations.push_back(owner);
  }

  // Send/receive data
  std::vector<uint> received_non_local_rows, received_non_local_cols;
  std::vector<double> received_non_local_vals;
  dolfin::MPI::distribute(send_non_local_rows, destinations, received_non_local_rows);
  dolfin::MPI::distribute(send_non_local_cols, destinations, received_non_local_cols);
  dolfin::MPI::distribute(send_non_local_vals, destinations, received_non_local_vals);

  assert(received_non_local_rows.size() == received_non_local_cols.size());
  assert(received_non_local_rows.size() == received_non_local_vals.size());

  // Add/insert off-process data
  for (uint i = 0; i < received_non_local_rows.size(); ++i)
  {
    dolfin_assert(received_non_local_rows[i] < _local_range.second && received_non_local_rows[i] >= _local_range.first);
    const uint I_local = received_non_local_rows[i] - _local_range.first;
    assert(I_local < this->_cols.size());
    assert(I_local < this->_vals.size());

    std::vector<uint>& rcols = this->_cols[I_local];
    std::vector<double>& rvals = this->_vals[I_local];
    const uint J = received_non_local_cols[i];

    // Check if column entry exists and insert
    const std::vector<uint>::const_iterator column = std::find(rcols.begin(), rcols.end(), J);
    if (column != rcols.end())
      rvals[column - rcols.begin()] += received_non_local_vals[i];
    else
    {
      rcols.push_back(J);
      rvals.push_back(received_non_local_vals[i]);
    }
  }
}
//-----------------------------------------------------------------------------
double STLMatrix::norm(std::string norm_type) const
{
  if (norm_type != "frobenius")
    error("Do not know to comput %s norm for STLMatrix", norm_type.c_str());

  double _norm = 0.0;
  for (uint i = 0; i < _vals.size(); ++i)
  {
    for (uint j = 0; j < _vals[i].size(); ++j)
      _norm += _vals[i][j]*_vals[i][j];
  }
  return std::sqrt(dolfin::MPI::sum(_norm));
}
//-----------------------------------------------------------------------------
void STLMatrix::getrow(uint row, std::vector<uint>& columns,
                       std::vector<double>& values) const
{
  dolfin_assert(row < _local_range.second && row >= _local_range.first);
  const uint local_row = row - _local_range.first;
  columns = this->_cols[local_row];
  values  = this->_vals[local_row];
}
//-----------------------------------------------------------------------------
void STLMatrix::ident(uint m, const uint* rows)
{
  std::pair<uint, uint> row_range = local_range(0);
  for (uint i = 0; i < m; ++i)
  {
    const uint global_row = rows[i];
    if (global_row >= row_range.first && global_row < row_range.second)
    {
      const uint local_row = global_row - row_range.first;
      std::vector<uint>& rcols   = this->_cols[local_row];
      std::vector<double>& rvals = this->_vals[local_row];

      // Zero row
      std::fill(rvals.begin(), rvals.end(), 0.0);

      // Place one on diagonal
      std::vector<uint>::const_iterator column = std::find(rcols.begin(), rcols.end(), global_row);
      if (column != rcols.end())
        rvals[column - rcols.begin()] = 1.0;
      else
      {
        rcols.push_back(global_row);
        rvals.push_back(1.0);
      }
    }
  }
}
//-----------------------------------------------------------------------------
std::string STLMatrix::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (uint i = 0; i < _local_range.second - _local_range.first; i++)
    {
      const std::vector<uint>& rcols = this->_cols[i];
      const std::vector<double>& rvals = this->_vals[i];

      std::stringstream line;
      line << std::setiosflags(std::ios::scientific);
      line << std::setprecision(16);

      line << "|";
      for (uint k = 0; k < rcols.size(); k++)
        line << " (" << i << ", " << rcols[k] << ", " << rvals[k] << ")";
      line << " |";

      s << line.str().c_str() << std::endl;
    }
  }
  else
  {
    s << "<STLMatrix of size " << size(0) << " x " << size(1) << ">";
  }
  return s.str();
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& STLMatrix::factory() const
{
  return STLFactory::instance();
}
//-----------------------------------------------------------------------------
dolfin::uint STLMatrix::nnz() const
{
  return dolfin::MPI::sum(local_nnz());
}
//-----------------------------------------------------------------------------
dolfin::uint STLMatrix::local_nnz() const
{
  uint _nnz = 0;
  for (uint i = 0; i < _cols.size(); ++i)
    _nnz += _cols[i].size();
  return _nnz;
}
//-----------------------------------------------------------------------------
void STLMatrix::csr(std::vector<double>& vals, std::vector<uint>& cols,
                    std::vector<uint>& row_ptr,
                    std::vector<uint>& local_to_global_row,
                    bool base_one) const
{
  // Reset data structures
  vals.clear();
  cols.clear();
  row_ptr.clear();
  local_to_global_row.clear();

  // Build CSR data structures
  if (base_one)
    row_ptr.push_back(1);
  else
    row_ptr.push_back(0);
  for (uint local_row = 0; local_row < _cols.size(); ++local_row)
  {
    vals.insert(vals.end(), _vals[local_row].begin(), _vals[local_row].end());
    cols.insert(cols.end(), _cols[local_row].begin(), _cols[local_row].end());

    row_ptr.push_back(row_ptr.back() + _cols[local_row].size());
    local_to_global_row.push_back(_local_range.first + local_row);
  }

  // Shift to array base 1
  if (base_one)
  {
    for (uint i = 0; i < local_to_global_row.size(); ++i)
      local_to_global_row[i]++;
    for (uint i = 0; i < cols.size(); ++i)
      cols[i]++;
  }
}
//-----------------------------------------------------------------------------
