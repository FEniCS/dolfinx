// Copyright (C) 2007-2011 Anders Logg
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
// Modified by Garth N. Wells, 2007-2009.
// Modified by Ola Skavhaug, 2007.
//
// First added:  2007-03-15
// Last changed: 2011-01-24

#ifndef __SCALAR_H
#define __SCALAR_H

#include <numeric>
#include <vector>
#include <dolfin/common/MPI.h>
#include "uBLASFactory.h"
#include "GenericTensor.h"

namespace dolfin
{

  class GenericSparsityPattern;

  /// This class represents a real-valued scalar quantity and
  /// implements the GenericTensor interface for scalars.

  class Scalar : public GenericTensor
  {
  public:

    /// Create zero scalar
    Scalar() : GenericTensor(), value(0.0)
    {}

    /// Destructor
    virtual ~Scalar()
    {}

    //--- Implementation of the GenericTensor interface ---

    /// Resize tensor to given dimensions
    virtual void resize(uint rank, const uint* dims)
    { assert(rank == 0); value = 0.0; }

    /// Initialize zero tensor using sparsity pattern
    void init(const GenericSparsityPattern& sparsity_pattern)
    { value = 0.0; }

    /// Return copy of tensor
    virtual Scalar* copy() const
    { Scalar* s = new Scalar(); s->value = value; return s; }

    /// Return tensor rank (number of dimensions)
    uint rank() const
    { return 0; }

    /// Return size of given dimension
    uint size(uint dim) const
    { error("The size() function is not available for scalars."); return 0; }

    /// Return local ownership range
    virtual std::pair<uint, uint> local_range(uint dim) const
    {
      error("The size() function is not available for scalars.");
      return std::make_pair(0, 0);
    }


    /// Get block of values
    void get(double* block, const uint* num_rows, const uint * const * rows) const
    { block[0] = value; }

    /// Set block of values
    void set(const double* block, const uint* num_rows, const uint * const * rows)
    { value = block[0]; }

    /// Add block of values
    void add(const double* block, const uint* num_rows, const uint * const * rows)
    {
      assert(block);
      value += block[0];
    }

    /// Add block of values
    void add(const double* block, const std::vector<const std::vector<uint>* >& rows)
    {
      assert(block);
      value += block[0];
    }

    /// Add block of values
    void add(const double* block, const std::vector<std::vector<uint> >& rows)
    {
      assert(block);
      value += block[0];
    }

    /// Set all entries to zero and keep any sparse structure
    void zero()
    { value = 0.0; }

    /// Finalize assembly of tensor
    void apply(std::string mode)
    {
      if (MPI::num_processes() > 1)
      {
        // Get values from other processes
        std::vector<double> values(MPI::num_processes());
        values[MPI::process_number()] = value;
        MPI::gather(values);

        // Sum contribution from each process
        value = std::accumulate(values.begin(), values.end(), 0.0);
      }
    }

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const
    {
      std::stringstream s;
      s << "<Scalar value " << value << ">";
      return s.str();
    }

    //--- Scalar interface ---

    /// Cast to real
    operator double() const
    { return value; }

    /// Assignment from real
    const Scalar& operator= (double value)
    { this->value = value; return *this; }

    //--- Special functions

    /// Return a factory for the default linear algebra backend
    inline LinearAlgebraFactory& factory() const
    { return dolfin::uBLASFactory<>::instance(); }

    /// Get value
    double getval() const
    { return value; }

  private:

    // Value of scalar
    double value;

  };

}

#endif
