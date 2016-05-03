// Copyright (C) 2007-2014 Anders Logg
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
// Modified by Garth N. Wells, 2007-2015.
// Modified by Ola Skavhaug, 2007.
// Modified by Martin Alnaes, 2014.

#ifndef __SCALAR_H
#define __SCALAR_H

#include <string>
#include <vector>
#include <dolfin/common/MPI.h>
#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/common/types.h>
#include "DefaultFactory.h"
#include "GenericTensor.h"
#include "TensorLayout.h"

namespace dolfin
{

  class TensorLayout;

  /// This class represents a real-valued scalar quantity and
  /// implements the GenericTensor interface for scalars.

  class Scalar : public GenericTensor
  {
  public:

    /// Create zero scalar
    Scalar() : Scalar(MPI_COMM_WORLD) {}

    /// Create zero scalar
    Scalar(MPI_Comm comm) : GenericTensor(), _value(0.0), _local_increment(0.0),
      _mpi_comm(comm)
      { SubSystemsManager::init_mpi(); }

    /// Destructor
    virtual ~Scalar() {}

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const TensorLayout& tensor_layout)
    {
      _value = 0.0;
      _local_increment = 0.0;
      _mpi_comm = tensor_layout.mpi_comm();
    }

    /// Return true if empty
    virtual bool empty() const
    { return false; }

    /// Return tensor rank (number of dimensions)
    virtual std::size_t rank() const
    { return 0; }

    /// Return size of given dimension
    virtual std::size_t size(std::size_t dim) const
    {
      // TODO: This is inconsistent in two ways:
      // - tensor.size(i) is defined for i < tensor.rank(), so not at all for a Scalar.
      // - the number of components of a tensor is the product of the sizes, returning 0 here makes no sense.
      // Is this used for anything? If yes, consider fixing that code. If no, just make this an error for any dim.

      if (dim != 0)
      {
        dolfin_error("Scalar.h",
                     "get size of scalar",
                     "Dim must be equal to zero.");
      }

      return 0;
    }

    /// Return local ownership range
    virtual std::pair<std::size_t, std::size_t>
      local_range(std::size_t dim) const
    {
      dolfin_error("Scalar.h",
                   "get local range of scalar",
                   "The local_range() function is not available for scalars");
      return std::make_pair(0, 0);
    }

    /// Get block of values
    virtual void get(double* block, const dolfin::la_index* num_rows,
             const dolfin::la_index * const * rows) const
    {
      dolfin_error("Scalar.h",
                   "get global value of scalar",
                   "The get() function is not available for scalars");
    }

    /// Set block of values using global indices
    virtual void set(const double* block, const dolfin::la_index* num_rows,
             const dolfin::la_index * const * rows)
    {
      dolfin_error("Scalar.h",
                   "set global value of scalar",
                   "The set() function is not available for scalars");
    }

    /// Set block of values using local indices
    virtual void set_local(const double* block, const dolfin::la_index* num_rows,
                   const dolfin::la_index * const * rows)
    {
      dolfin_error("Scalar.h",
                   "set local value of scalar",
                   "The set_local() function is not available for scalars");
    }

    /// Add block of values using global indices
    virtual void add(const double* block, const dolfin::la_index* num_rows,
             const dolfin::la_index * const * rows)
    {
      dolfin_assert(block);
      _local_increment += block[0];
    }

    /// Add block of values using local indices
    virtual void add_local(const double* block, const dolfin::la_index* num_rows,
                   const dolfin::la_index * const * rows)
    {
      dolfin_assert(block);
      _local_increment += block[0];
    }

    /// Add block of values using global indices
    virtual void add(const double* block,
             const std::vector<ArrayView<const dolfin::la_index>>& rows)
    {
      dolfin_assert(block);
      _local_increment += block[0];
    }

    /// Add block of values using local indices
    virtual void add_local(const double* block,
             const std::vector<ArrayView<const dolfin::la_index>>& rows)
    {
      dolfin_assert(block);
      _local_increment += block[0];
    }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero()
    {
      _value = 0.0;
      _local_increment = 0.0;
    }

    /// Finalize assembly of tensor
    virtual void apply(std::string mode)
    {
      _value = _value + MPI::sum(_mpi_comm, _local_increment);
      _local_increment = 0.0;
    }

    /// Return MPI communicator
    virtual MPI_Comm mpi_comm() const
    { return _mpi_comm; }

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const
    {
      std::stringstream s;
      s << "<Scalar value " << _value << ">";
      return s.str();
    }

    //--- Scalar interface ---

    /// Return copy of scalar
    virtual std::shared_ptr<Scalar> copy() const
    {
      std::shared_ptr<Scalar> s(new Scalar);
      s->_value = _value;
      s->_local_increment = _local_increment;
      s->_mpi_comm = _mpi_comm;
      return s;
    }

    //--- Special functions

    /// Return a factory for the default linear algebra backend
    virtual GenericLinearAlgebraFactory& factory() const
    {
      DefaultFactory f;
      return f.factory();
    }

    /// Get final value (assumes prior apply(), not part of
    /// GenericTensor interface)
    double get_scalar_value() const
    { return _value; }

    /// Add to local increment (added for testing, remove if we add a
    /// better way from python)
    void add_local_value(double value)
    { _local_increment += value; }

  private:

    // Value of scalar
    double _value;

    // Local intermediate value of scalar prior to apply call
    double _local_increment;

    // MPI communicator
     MPI_Comm _mpi_comm;

  };

}

#endif
