// Copyright (C) 2007-2012 Anders Logg, Garth N. Wells, Ola Skavhaug,
// Martin Alnæs
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
// First added:  2012-08-22
// Last changed: 2012-08-24

#ifndef __LINEAR_ALGEBRA_OBJECT_H
#define __LINEAR_ALGEBRA_OBJECT_H

#include <memory>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  /// This is a common base class for all DOLFIN linear algebra
  /// objects. In particular, it provides casting mechanisms between
  /// different types.

  class LinearAlgebraObject : public virtual Variable
  {
  public:

    /// Cast object to its derived class, if possible (const version).
    /// An error is thrown if the cast is unsuccessful.
    template<typename T> const T& down_cast() const
    {
      try
      {
        return dynamic_cast<const T&>(*instance());
      }
      catch (std::exception& e)
      {
        dolfin_error("LinearAlgebraObject.h",
                     "down-cast linear algebra object to requested type",
                     "%s", e.what());
      }

      // Return something to keep the compiler happy, code will not be
      // reached
      return dynamic_cast<const T&>(*instance());
    }

    /// Cast object to its derived class, if possible (non-const version).
    /// An error is thrown if the cast is unsuccessful.
    template<typename T> T& down_cast()
    {
      try
      {
        return dynamic_cast<T&>(*instance());
      }
      catch (std::exception& e)
      {
        dolfin_error("LinearAlgebraObject.h",
                     "down-cast linear algebra object to requested type",
                     "%s", e.what());
      }

      // Return something to keep the compiler happy, code will not be reached
      return dynamic_cast<T&>(*instance());
    }

    /// Cast shared pointer object to its derived class, if possible.
    /// Caller must check for success (returns null if cast fails).
    template<typename X, typename Y>
    static std::shared_ptr<X> down_cast(std::shared_ptr<Y> A)
    {
      // Try to down cast shared pointer
      std::shared_ptr<X> _matA = std::dynamic_pointer_cast<X>(A);

      // If down cast fails, try to get shared ptr instance to
      // unwrapped object
      if (!_matA)
      {
        // Try to get instance to unwrapped object and cast
        if (A->shared_instance())
          _matA = std::dynamic_pointer_cast<X>(A->shared_instance());
      }
      return _matA;
    }

    /// Return concrete instance / unwrap (const version)
    virtual const LinearAlgebraObject* instance() const
    { return this; }

    /// Return concrete instance / unwrap (non-const version)
    virtual LinearAlgebraObject* instance()
    { return this; }

    /// Return concrete shared ptr instance / unwrap (const version)
    virtual std::shared_ptr<const LinearAlgebraObject> shared_instance() const
    { return std::shared_ptr<const LinearAlgebraObject>(); }

    /// Return concrete shared ptr instance / unwrap (non-const version)
    virtual std::shared_ptr<LinearAlgebraObject> shared_instance()
    { return std::shared_ptr<LinearAlgebraObject>(); }

    /// Return MPI communicator
    virtual MPI_Comm mpi_comm() const = 0;

  };

  /// Cast object to its derived class, if possible (non-const version).
  /// An error is thrown if the cast is unsuccessful.
  template<typename Y, typename X>
  Y& as_type(X& x)
  {
    try
    {
      return dynamic_cast<Y&>(*x.instance());
    }
    catch (std::exception& e)
    {
      dolfin_error("LinearAlgebraObject.h",
                   "down-cast linear algebra object to requested type",
                   "%s", e.what());
    }

    // Return something to keep the compiler happy, code will not be reached
    return dynamic_cast<Y&>(*x.instance());
  }

  // This function has been copied from Boost 1.53.0
  // (boost/smart_ptr/shared_ptr.hpp) and modified. It has been modified
  // because the line
  //
  //     (void) dynamic_cast< T* >( static_cast< U* >( 0 ) );
  //
  // breaks with the Intel C++ compiler (icpc 13.0.1 20121010). This
  // modified function should only be called when using the Intel compiler
  // and compiler and Boost updates should be tested.
  #if defined __INTEL_COMPILER
  template<class T, class U>
  std::shared_ptr<T>
    dolfin_dynamic_pointer_cast(std::shared_ptr<U> const & r )
  {
      // Below give error with icpc 13.0.1 20121010
      //(void) dynamic_cast< T* >( static_cast< U* >( 0 ) );
      typedef typename std::shared_ptr<T>::element_type E;
      E * p = dynamic_cast< E* >(r.get());
      return p ? std::shared_ptr<T>(r, p) : std::shared_ptr<T>();
  }
  #endif

  /// Cast shared pointer object to its derived class, if possible.
  /// Caller must check for success (returns null if cast fails).
  template<typename Y, typename X>
  std::shared_ptr<Y> as_type(std::shared_ptr<X> x)
  {
    // Try to down cast shared pointer
    std::shared_ptr<Y> y = std::dynamic_pointer_cast<Y>(x);

    // If down cast fails, try to get shared ptr instance to unwrapped object
    if (!y)
    {
      // Try to get instance to unwrapped object and cast
      if (x->shared_instance())
      {
        // Called modified function if using Intel compiler. See comments
        // on above function.
        #if defined __INTEL_COMPILER
        y = dolfin_dynamic_pointer_cast<Y>(x->shared_instance());
        #else
        y = std::dynamic_pointer_cast<Y>(x->shared_instance());
        #endif
      }
    }
    return y;
  }

  /// Check whether the object matches a specific type
  template<typename Y, typename X>
  bool has_type(const X& x)
  {
    return bool(dynamic_cast<const Y*>(x.instance()));
  }

}

#endif
