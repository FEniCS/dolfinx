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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2007-2011.
// Modified by Ola Skavhaug, 2007.
// Modified by Martin Alnæs, 2008.
//
// First added:  2007-01-17
// Last changed: 2010-11-29

#ifndef __GENERIC_TENSOR_H
#define __GENERIC_TENSOR_H

#include <exception>
#include <typeinfo>
#include <boost/shared_ptr.hpp>
#include <dolfin/log/log.h>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class GenericSparsityPattern;
  class LinearAlgebraFactory;

  /// This class defines a common interface for arbitrary rank tensors.

  class GenericTensor : public virtual Variable
  {
  public:

    /// Destructor
    virtual ~GenericTensor() {}

    //--- Basic GenericTensor interface ---

    /// Return true if tensor is distributed
    virtual bool distributed() const = 0;

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const GenericSparsityPattern& sparsity_pattern) = 0;

    /// Return copy of tensor
    virtual GenericTensor* copy() const = 0;

    /// Return tensor rank (number of dimensions)
    virtual uint rank() const = 0;

    /// Return size of given dimension
    virtual uint size(uint dim) const = 0;

    /// Return local ownership range
    virtual std::pair<uint, uint> local_range(uint dim) const = 0;

    /// Get block of values
    virtual void get(double* block, const uint* num_rows,
                     const uint * const * rows) const = 0;

    /// Set block of values
    virtual void set(const double* block, const uint* num_rows,
                     const uint * const * rows) = 0;

    /// Add block of values
    virtual void add(const double* block,
                     const std::vector<const std::vector<uint>* >& rows) = 0;

    /// Add block of values
    virtual void add(const double* block,
                     const std::vector<std::vector<uint> >& rows) = 0;

    /// Add block of values
    virtual void add(const double* block, const uint* num_rows,
                     const uint * const * rows) = 0;

    /// Set all entries to zero and keep any sparse structure
    virtual void zero() = 0;

    /// Finalize assembly of tensor
    virtual void apply(std::string mode) = 0;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

    //--- Special functions, downcasting to concrete types ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const = 0;

    /// Cast a GenericTensor to its derived class (const version)
    template<class T> const T& down_cast() const
    {
      try
      {
        return dynamic_cast<const T&>(*instance());
      }
      catch (std::exception& e)
      {
        error("GenericTensor cannot be cast to the requested type: %s", e.what());
      }

      // Return something to keep the compiler happy. Code will never be reached.
      return dynamic_cast<const T&>(*instance());
    }

    /// Cast a GenericTensor to its derived class (non-const version)
    template<class T> T& down_cast()
    {
      try
      {
        return dynamic_cast<T&>(*instance());
      }
      catch (std::exception& e)
      {
        error("GenericTensor cannot be cast to the requested type: %s", e.what());
      }

      // Return something to keep the compiler happy. Code will never be reached.
      return dynamic_cast<T&>(*instance());
    }

    /// Cast a GenericTensor shared ptr to its derived class. Caller
    /// must check for success (returns null if cast fails).
    template<class X, class Y>
    static boost::shared_ptr<X> down_cast(const boost::shared_ptr<Y> A)
    {
      // Try to down cast shared pointer
      boost::shared_ptr<X> _A = boost::dynamic_pointer_cast<X>(A);

      // If down cast fails, try to get shared ptr instance to unwrapped object
      if (!_A)
      {
        // Try to get instance to unwrapped object and cast
        if (A->shared_instance())
          _A = boost::dynamic_pointer_cast<X>(A->shared_instance());
      }
      return _A;
    }

    /// Check whether the GenericTensor instance matches a specific type
    template<class T> bool has_type() const
    { return bool(dynamic_cast<const T*>(instance())); }

    //--- Special functions, intended for library use only ---

    /// Return concrete instance / unwrap (const version)
    virtual const GenericTensor* instance() const
    { return this; }

    /// Return concrete instance / unwrap (non-const version)
    virtual GenericTensor* instance()
    { return this; }

    /// Return concrete shared ptr instance / unwrap (const version)
    virtual boost::shared_ptr<const GenericTensor> shared_instance() const
    { return boost::shared_ptr<const GenericTensor>(); }

    /// Return concrete shared ptr instance / unwrap
    virtual boost::shared_ptr<GenericTensor> shared_instance()
    { return boost::shared_ptr<GenericTensor>(); }

    /// Assignment (must be overloaded by subclass)
    virtual const GenericTensor& operator= (const GenericTensor& x)
    { error("Assignment operator not implemented by subclass"); return *this; }

  };

}

#endif
