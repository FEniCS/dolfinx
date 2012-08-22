// Copyright (C) 2007-2012 Anders Logg, Garth N. Wells, Ola Skavhaug, Martin Alnæs
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
// Last changed: 2012-08-22

#ifndef __LINEAR_ALGEBRA_OBJECT_H
#define __LINEAR_ALGEBRA_OBJECT_H

#include <dolfin/common/Variable.h>

namespace dolfin
{

  /// This is a common base class for all DOLFIN linear algebra
  /// objects. In particular, it provides casting mechanisms between
  /// different types.

  // FIXME: Might be that inheritance from Variable needs to be virtual

  class LinearAlgebraObject : public Variable
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

      // Return something to keep the compiler happy, code will not be reached
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

    /// Check whether the object matches a specific type
    template<typename T> bool has_type() const
    { return bool(dynamic_cast<const T*>(instance())); }

    /// Return concrete instance / unwrap (const version)
    virtual const LinearAlgebraObject* instance() const
    { return this; }

    /// Return concrete instance / unwrap (non-const version)
    virtual LinearAlgebraObject* instance()
    { return this; }

    /// Return concrete shared ptr instance / unwrap (const version)
    virtual boost::shared_ptr<const LinearAlgebraObject> shared_instance() const
    { return boost::shared_ptr<const LinearAlgebraObject>(); }

    /// Return concrete shared ptr instance / unwrap (non-const version)
    virtual boost::shared_ptr<LinearAlgebraObject> shared_instance()
    { return boost::shared_ptr<LinearAlgebraObject>(); }

  };

}

#endif
