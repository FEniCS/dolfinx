// Copyright (C) 2015 Garth N. Wells
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

#ifndef __DOLFIN_ARRAYVIEW_H
#define __DOLFIN_ARRAYVIEW_H

#include <cstddef>
#include <dolfin/log/log.h>

namespace dolfin
{

  /// This class provides a wrapper for a pointer to an array. It does
  /// not own the data, and will not be valid if the underlying data
  /// goes out-of-scope.

  template <typename T> class ArrayView
  {

  public:

    /// Construct array from a pointer. Array does not take ownership.
    ArrayView(std::size_t N, T* x) : _size(N), _x(x) {}

    /// Copy constructor
    ArrayView(const Array& x) : _size(x._size), _x(x._x) {}

    /// Destructor
    ~Array() {}

    /// Return size of array
    std::size_t size() const
    { return _size; }

    /// Access value of given entry (const version)
    const T& operator[] (std::size_t i) const
    { dolfin_assert(i < _size); return _x[i]; }

    /// Access value of given entry (non-const version)
    T& operator[] (std::size_t i)
    { dolfin_assert(i < _size); return _x[i]; }

    /// Return pointer to data (const version)
    const T* data() const
    { return _x; }

    /// Return pointer to data (non-const version)
    T* data()
    { return _x; }

  private:

    /// Length of array
    const std::size_t _size;

    /// Array data
    T* _x;

  };

}

#endif
