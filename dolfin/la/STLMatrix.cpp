// Copyright (C) 2007-2008 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Garth N. Wells, 2007.
// Modified by Ilmar Wilbers, 2008.
//
// First added:  2007-01-17
// Last changed: 2010-11-08

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "dolfin/common/Timer.h"

#include "STLFactory.h"
#include "STLMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void STLMatrix::add(const double* block, uint m, const uint* rows, uint n, const uint* cols)
{
  // Perform a simple linear search along each column. Otherwise,
  // append the value (calling push_back).

  // Map-based storage
  /*
  uint pos = 0;
  for (uint i = 0; i < m; i++)
  {
    const uint I = rows[i];
    for (uint j = 0; j < n; j++)
    {
      const uint J = cols[j];
      matrix[I][J] += block[pos++];
    }
  }
  */

  // Iterate over rows
  uint pos = 0;
  for (uint i = 0; i < m; i++)
  {
    const uint I = rows[i];
    std::vector<uint>& rcols = this->cols[I];
    std::vector<double>& rvals = this->vals[I];

    // Iterate over columns
    for (uint j = 0; j < n; j++)
    {
      const uint J = cols[j];

      // Check if column entry exists and insert
      const std::vector<uint>::const_iterator column = std::find(rcols.begin(), rcols.end(), J);
      if (column != rcols.end())
        rvals[column - rcols.begin()] += block[pos++];
      else
      {
        rcols.push_back(J);
        rvals.push_back(block[pos++]);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void STLMatrix::apply(std::string mode)
{
  // Experimental code for copying a vectpr<map> row-by-row in a compressed
  // format

  /*
  tic();
  for (uint i = 0; i < matrix.size(); ++i)
  {
    //const std::map<uint, double>& row = matrix[i];
    const boost::unordered_map<uint, double>& row = matrix[i];

    std::vector<double> this_row(row.size());
    std::vector<uint> this_row_index(row.size());

    uint pos = 0;
    //for (std::map<uint, double>::const_iterator entry = row.begin(); entry != row.end(); ++entry)
    for (boost::unordered_map<uint, double>::const_iterator entry = row.begin(); entry != row.end(); ++entry)
    {
      this_row[pos]   = entry->second;
      this_row_index[pos++] = entry->first;
    }
  }
  const double time = toc();
  std::cout << "STLMatrix apply time: " << time << std::endl;
  */
}
//-----------------------------------------------------------------------------
std::string STLMatrix::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    for (uint i = 0; i < dims[0]; i++)
    {
      const std::vector<uint>& rcols = this->cols[i];
      const std::vector<double>& rvals = this->vals[i];

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
