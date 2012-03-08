// Copyright (C) 2009-2011 Garth N. Wells
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
// Last changed: 2012-02-23

#ifndef __DOLFIN_ARRAY_H
#define __DOLFIN_ARRAY_H

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <dolfin/common/constants.h>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>

#include "NoDeleter.h"

namespace dolfin
{

  /// This class provides a simple wrapper for a pointer to an array. A purpose
  /// of this class is to enable the simple and safe exchange of data between
  /// C++ and Python.

  template <typename T> class Array
  {

  public:

    /// Create an empty array, with ownership. Must be resized or assigned to to be of any use.
    explicit Array() : _size(0), _x(NULL), _owner(true) {}

    /// Create array of size N. Array has ownership.
    explicit Array(uint N) : _size(N), _x(new T[N]), _owner(true) {}

    /// Create an array wrapping a std::vector. Array does not take ownership.
    explicit Array(std::vector<T>& vec) : _size(vec.size()), _x(&vec[0]), _owner(false) {}

    /// Construct array from a pointer. Array does not take ownership.
    Array(uint N, T* x) : _size(N), _x(x), _owner(false) {}

    /// Destructor.
    ~Array()
    {
      if (_owner && _x)
        delete[] _x;
    }

    /// Resize the array. If the new size if different from the old size, the
    /// Array must be owner; the old contents are then lost.
    void resize(uint N)
    {
      if (_size == N)
        return;
      if (!_owner)
        dolfin_error("Array.h", "resize", "Only owned arrays can be resized");
      if (_x)
        delete[] _x;
      _x = (N == 0 ? NULL : new T[N]);
      _size = N;
    }

    /// Clear the array (resize to 0).
    void clear()
    {
      resize(0);
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

    /// Set entries which meet (abs(x[i]) < eps) to zero
    void zero_eps(double eps=DOLFIN_EPS);

    /// Access value of given entry (const version)
    const T& operator[] (uint i) const
    { dolfin_assert(i < _size); return _x[i]; }

    /// Access value of given entry (non-const version)
    T& operator[] (uint i)
    { dolfin_assert(i < _size); return _x[i]; }

    /// Assignment operator. If resize is required, the Array must be owner.
    Array& operator= (const Array& other)
    {
      resize(other._size);
      if (_size > 0)
        std::copy(&other._x[0], &other._x[_size], &_x[0]);
      return *this;
    }

    /// Assignment operator from std::vector. If resize is required, the Array must be owner.
    Array& operator= (const std::vector<T>& other)
    {
      resize(other.size());
      if (_size > 0)
        std::copy(&other[0], &other[_size], &_x[0]);
      return *this;
    }

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
    uint _size;

    /// Array data
    T* _x;

    /// True if instance is owner of data
    bool _owner;
  };

  template <typename T>
  inline void Array<T>::zero_eps(double eps)
  {
    dolfin_error("Array.h",
                 "zero small entries in array",
                 "Only available when data type is <double>");
  }

  template <>
  inline void Array<double>::zero_eps(double eps)
  {
    for (uint i = 0; i < _size; ++i)
    {
      if (std::abs(_x[i]) < eps)
        _x[i] = 0.0;
    }
  }

}

#endif
