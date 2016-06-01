// Copyright (C) 2007-2012 Anders Logg
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
// Modified by Garth N. Wells 2007-2011
// Modified by Ola Skavhaug 2007
// Modified by Martin Alnæs 2008
//
// First added:  2007-01-17
// Last changed: 2012-08-22

#ifndef __GENERIC_TENSOR_H
#define __GENERIC_TENSOR_H

#include <cstdint>
#include <exception>
#include <memory>
#include <typeinfo>
#include <utility>

#include <dolfin/common/ArrayView.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>
#include "LinearAlgebraObject.h"

namespace dolfin
{

  class TensorLayout;
  class GenericLinearAlgebraFactory;

  /// This class defines a common interface for arbitrary rank tensors.

  class GenericTensor : public virtual LinearAlgebraObject
  {
  public:

    /// Destructor
    virtual ~GenericTensor() {}

    //--- Basic GenericTensor interface ---

    /// Initialize zero tensor using tensor layout
    virtual void init(const TensorLayout& tensor_layout) = 0;

    /// Return true if empty
    virtual bool empty() const = 0;

    /// Return tensor rank (number of dimensions)
    virtual std::size_t rank() const = 0;

    /// Return size of given dimension
    virtual std::size_t size(std::size_t dim) const = 0;

    /// Return local ownership range
    virtual std::pair<std::int64_t, std::int64_t>
    local_range(std::size_t dim) const = 0;

    /// Get block of values
    virtual void get(double* block, const dolfin::la_index* num_rows,
                     const dolfin::la_index * const * rows) const = 0;

    /// Set block of values using global indices
    virtual void set(const double* block, const dolfin::la_index* num_rows,
                     const dolfin::la_index * const * rows) = 0;

    /// Set block of values using local indices
    virtual void set_local(const double* block,
                           const dolfin::la_index* num_rows,
                           const dolfin::la_index * const * rows) = 0;

    /// Add block of values using global indices
    virtual
      void add(const double* block,
           const std::vector<ArrayView<const dolfin::la_index>>& rows) = 0;

    /// Add block of values using local indices
    virtual void add_local(
      const double* block,
      const std::vector<ArrayView<const dolfin::la_index>>& rows) = 0;


    /// Add block of values using global indices
    virtual void add(const double* block, const dolfin::la_index* num_rows,
                     const dolfin::la_index * const * rows) = 0;

    /// Add block of values using local indices
    virtual void add_local(const double* block,
                           const dolfin::la_index* num_rows,
                           const dolfin::la_index * const * rows) = 0;

    /// Set all entries to zero and keep any sparse structure
    virtual void zero() = 0;

    /// Finalize assembly of tensor
    virtual void apply(std::string mode) = 0;

    /// Return MPI communicator
    //virtual MPI_Comm mpi_comm() const = 0;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

    /// Return linear algebra backend factory
    virtual GenericLinearAlgebraFactory& factory() const = 0;

  };

}

#endif
