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
        y = std::dynamic_pointer_cast<Y>(x->shared_instance());
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
