// Copyright (C) 2006-2010 Garth N. Wells
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
// Modified by Anders Logg 2006-2011
// Modified by Kent-Andre Mardal 2008
// Modified by Ola Skavhaug 2008
// Modified by Martin Sandve Alnes 2009
// Modified by Johan Hake 2009-2010
//
// First added:  2006-04-25
// Last changed: 2011-11-11

#ifndef __GENERIC_VECTOR_H
#define __GENERIC_VECTOR_H

#include <algorithm>
#include <utility>
#include <vector>
#include <dolfin/common/ArrayView.h>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>
#include "IndexMap.h"
#include "TensorLayout.h"
#include "GenericTensor.h"

namespace dolfin
{
  template<typename T> class Array;

  /// This class defines a common interface for vectors.

  class GenericVector : public GenericTensor
  {
  public:

    /// Destructor
    virtual ~GenericVector() {}

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern
    /// FIXME: This needs to be implemented on backend side! Remove it!
    virtual void init(const TensorLayout& tensor_layout)
    {
      if (!empty())
      {
        dolfin_error("GenericVector.h",
                     "initialize vector",
                     "Vector cannot be initialised more than once");
      }

      std::vector<dolfin::la_index> ghosts;
      std::vector<std::size_t> local_to_global(tensor_layout.index_map(0)->size(IndexMap::MapSize::ALL));

      // FIXME: should just pass index_map to init()
      for (std::size_t i = 0; i != local_to_global.size(); ++i)
        local_to_global[i] = tensor_layout.index_map(0)->local_to_global(i);

      // FIXME: temporary hack - needs passing tensor layout directly to backend
      if (tensor_layout.is_ghosted() == TensorLayout::Ghosts::GHOSTED)
      {
        const std::size_t nowned
          = tensor_layout.index_map(0)->size(IndexMap::MapSize::OWNED);
        const std::size_t nghosts
          = tensor_layout.index_map(0)->size(IndexMap::MapSize::UNOWNED);
        ghosts.resize(nghosts);
        for (std::size_t i = 0; i != nghosts; ++i)
          ghosts[i] = local_to_global[i + nowned];
      }

      init(tensor_layout.mpi_comm(), tensor_layout.local_range(0),
           local_to_global, ghosts);
      zero();
    }

    /// Return tensor rank (number of dimensions)
    virtual std::size_t rank() const
    { return 1; }

    /// Return size of given dimension
    virtual std::size_t size(std::size_t dim) const
    { dolfin_assert(dim == 0); return size(); }

    /// Return local ownership range
    virtual std::pair<std::size_t, std::size_t>
    local_range(std::size_t dim) const
    { dolfin_assert(dim == 0); return local_range(); }

    /// Get block of values using global indices
    virtual void get(double* block, const dolfin::la_index* num_rows,
                     const dolfin::la_index * const * rows) const
    { get(block, num_rows[0], rows[0]); }

    /// Get block of values using local indices
    virtual void get_local(double* block, const dolfin::la_index* num_rows,
                           const dolfin::la_index * const * rows) const
    { get_local(block, num_rows[0], rows[0]); }

    /// Set block of values using global indices
    virtual void set(const double* block, const dolfin::la_index* num_rows,
                     const dolfin::la_index * const * rows)
    { set(block, num_rows[0], rows[0]); }

    /// Set block of values using local indices
    virtual void set_local(const double* block,
                           const dolfin::la_index* num_rows,
                           const dolfin::la_index * const * rows)
    { set_local(block, num_rows[0], rows[0]); }

    /// Add block of values using global indices
    virtual void add(const double* block, const dolfin::la_index* num_rows,
                     const dolfin::la_index * const * rows)
    { add(block, num_rows[0], rows[0]); }

    /// Add block of values using local indices
    virtual void add_local(const double* block,
                           const dolfin::la_index* num_rows,
                           const dolfin::la_index * const * rows)
    { add_local(block, num_rows[0], rows[0]); }

    /// Add block of values using global indices
    virtual void
      add(const double* block,
          const std::vector<ArrayView<const dolfin::la_index>>& rows)
    { add(block, rows[0].size(), rows[0].data()); }

    /// Add block of values using local indices
    virtual void
    add_local(const double* block,
              const std::vector<ArrayView<const dolfin::la_index>>& rows)
    { add_local(block, rows[0].size(), rows[0].data()); }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero() = 0;

    /// Finalize assembly of tensor
    virtual void apply(std::string mode) = 0;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

    //--- Vector interface ---

    /// Return copy of vector
    virtual std::shared_ptr<GenericVector> copy() const = 0;

    /// Initialize vector to global size N
    virtual void init(MPI_Comm comm, std::size_t N) = 0;

    /// Initialize vector with given local ownership range
    virtual void init(MPI_Comm comm,
                      std::pair<std::size_t, std::size_t> range) = 0;

    /// Initialise vector with given ownership range and with ghost
    /// values
    /// FIXME: Reimplement using init(const TensorLayout&) and deprecate
    virtual void init(MPI_Comm comm,
                      std::pair<std::size_t, std::size_t> range,
                      const std::vector<std::size_t>& local_to_global_map,
                      const std::vector<la_index>& ghost_indices) = 0;

    /// Return global size of vector
    virtual std::size_t size() const = 0;

    /// Return local size of vector
    virtual std::size_t local_size() const = 0;

    /// Return local ownership range of a vector
    virtual std::pair<std::size_t, std::size_t> local_range() const = 0;

    /// Determine whether global vector index is owned by this process
    virtual bool owns_index(std::size_t i) const = 0;

    /// Get block of values using global indices (values must all live
    /// on the local process, ghosts cannot be accessed)
    virtual void get(double* block, std::size_t m,
                     const dolfin::la_index* rows) const = 0;

    /// Get block of values using local indices (values must all live
    /// on the local process, ghost are accessible)
    virtual void get_local(double* block, std::size_t m,
                           const dolfin::la_index* rows) const = 0;

    /// Set block of values using global indices
    virtual void set(const double* block, std::size_t m,
                     const dolfin::la_index* rows) = 0;

    /// Set block of values using local indices
    virtual void set_local(const double* block, std::size_t m,
                           const dolfin::la_index* rows) = 0;

    /// Add block of values using global indices
    virtual void add(const double* block, std::size_t m,
                     const dolfin::la_index* rows) = 0;

    /// Add block of values using local indices
    virtual void add_local(const double* block, std::size_t m,
                           const dolfin::la_index* rows) = 0;

    /// Get all values on local process
    virtual void get_local(std::vector<double>& values) const = 0;

    /// Set all values on local process
    virtual void set_local(const std::vector<double>& values) = 0;

    /// Add values to each entry on local process
    virtual void add_local(const Array<double>& values) = 0;

    /// Gather entries into local vector x
    virtual void gather(GenericVector& x,
                        const std::vector<dolfin::la_index>& indices) const = 0;

    /// Gather entries into x
    virtual void gather(std::vector<double>& x,
                        const std::vector<dolfin::la_index>& indices) const = 0;

    /// Gather all entries into x on process 0
    virtual void gather_on_zero(std::vector<double>& x) const = 0;

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(double a, const GenericVector& x) = 0;

    /// Replace all entries in the vector by their absolute values
    virtual void abs() = 0;

    /// Return inner product with given vector
    virtual double inner(const GenericVector& x) const = 0;

    /// Return norm of vector
    virtual double norm(std::string norm_type) const = 0;

    /// Return minimum value of vector
    virtual double min() const = 0;

    /// Return maximum value of vector
    virtual double max() const = 0;

    /// Return sum of vector
    virtual double sum() const = 0;

    /// Return sum of selected rows in vector. Repeated entries are
    /// only summed once.
    virtual double sum(const Array<std::size_t>& rows) const = 0;

    /// Multiply vector by given number
    virtual const GenericVector& operator*= (double a) = 0;

    /// Multiply vector by another vector pointwise
    virtual const GenericVector& operator*= (const GenericVector& x) = 0;

    /// Divide vector by given number
    virtual const GenericVector& operator/= (double a) = 0;

    /// Add given vector
    virtual const GenericVector& operator+= (const GenericVector& x) = 0;

    /// Add number to all components of a vector
    virtual const GenericVector& operator+= (double a) = 0;

    /// Subtract given vector
    virtual const GenericVector& operator-= (const GenericVector& x) = 0;

    /// Subtract number from all components of a vector
    virtual const GenericVector& operator-= (double a) = 0;

    /// Assignment operator
    virtual const GenericVector& operator= (const GenericVector& x) = 0;

    /// Assignment operator
    virtual const GenericVector& operator= (double a) = 0;

    //--- Convenience functions ---

    /// Get value of given entry
    virtual double operator[] (dolfin::la_index i) const
    { double value(0); get_local(&value, 1, &i); return value; }

    /// Get value of given entry
    virtual double getitem(dolfin::la_index i) const
    { double value(0); get_local(&value, 1, &i); return value; }

    /// Set given entry to value. apply("insert") should be called
    /// before using using the object.
    virtual void setitem(dolfin::la_index i, double value)
    { set(&value, 1, &i); }

  };

}

#endif
