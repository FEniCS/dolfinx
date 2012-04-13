// Copyright (C) 2009-2012 Garth N. Wells
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
// Modified by Anders Logg 2010-2011
// Modified by Joachim B Haga 2012
//
// First added:  2009-12-06
// Last changed: 2012-03-12

#ifndef __DOLFIN_ARRAY_H
#define __DOLFIN_ARRAY_H

#include <sstream>
#include <string>

#include <dolfin/common/constants.h>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  /// This class provides a simple wrapper for a pointer to an array. A
  /// purpose of this class is to enable the simple and safe exchange
  /// of data between C++ and Python.

  template <typename T> class Array
  {

  public:

    /// Create array of size N. Array has ownership.
    explicit Array(uint N) : _size(N), _x(new T[N]), _owner(true) {}

    /// Construct array from a pointer. Array does not take ownership.
    Array(uint N, T* x) : _size(N), _x(x), _owner(false) {}

    /// Destructor
    ~Array()
    {
      if (_owner)
        delete [] _x;
    }

    /// Return informal string representation (pretty-print).
    /// Note that the Array class is not a subclass of Variable (for
    /// efficiency) which means that one needs to call str() directly
    /// instead of using the info() function on Array objects.
    std::string str(bool verbose) const
    {
      std::stringstream s;

      if (verbose)
      {
        s << str(false) << std::endl << std::endl;

        for (uint i = 0; i < size(); i++)
          s << i << ": " << (*this)[i] << std::endl;
      }
      else
        s << "<Array<T> of size " << size() << ">";

      return s.str();
    }

    /// Return size of array
    uint size() const
    { return _size; }

    /// Access value of given entry (const version)
    const T& operator[] (uint i) const
    { dolfin_assert(i < _size); return _x[i]; }

    /// Access value of given entry (non-const version)
    T& operator[] (uint i)
    { dolfin_assert(i < _size); return _x[i]; }

    /// Return pointer to data (const version)
    const T* data() const
    { return _x; }

    /// Return pointer to data (non-const version)
    T* data()
    { return _x; }

   private:

    /// Disable copy construction, to avoid unanticipated sharing or
    /// copying of data. This means that an Array must always be passed as
    /// reference, or as a (possibly shared) pointer.
    Array(const Array& other) /* leave body undefined */;

  private:

    /// Length of array
    const uint _size;

    /// Array data
    T* _x;

    /// True if instance is owner of data
    bool _owner;

  };

}

#endif
