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

  /// This class provides a wrapper for a pointer to an array. It
  /// never owns the data, and will not be valid if the underlying
  /// data goes out-of-scope.

  template <typename T> class ArrayView
  {

  public:

    /// Constructor
    ArrayView() : _size(0), _x(NULL) {}

    /// Construct array from a pointer. Array does not take ownership.
    ArrayView(std::size_t N, T* x) : _size(N), _x(x) {}

    /// Construct array from a container with the the data() and
    /// size() functions
    template<typename V>
      explicit ArrayView(V& v) : _size(v.size()), _x(v.data()) {}

    /// Copy constructor
    ArrayView(const ArrayView& x) : _size(x._size), _x(x._x) {}

    /// Destructor
    ~ArrayView() {}

    /// Update object to point to new data
    void set(std::size_t N, T* x)
    { _size = N; _x = x; }

    /// Update object to point to new container
    template<typename V>
    void set(V& v)
    { _size = v.size(); _x = v.data(); }

    /// Return size of array
    std::size_t size() const
    { return _size; }

    /// Test if array view is empty
    bool empty() const
    { return (_size == 0) ? true : false; }

    /// Access value of given entry (const version)
    const T& operator[] (std::size_t i) const
    { dolfin_assert(i < _size); return _x[i]; }

    /// Access value of given entry (non-const version)
    T& operator[] (std::size_t i)
    { dolfin_assert(i < _size); return _x[i]; }

    /// Pointer to start of array
    T* begin()
    { return &_x[0]; }

    /// Pointer to start of array (const)
    const T* begin() const
    { return &_x[0]; }

    /// Pointer to beyond end of array
    T* end()
    { return &_x[_size]; }

    /// Pointer to beyond end of array (const)
    const T* end() const
    { return &_x[_size]; }

    /// Return pointer to data (const version)
    const T* data() const
    { return _x; }

    /// Return pointer to data (non-const version)
    T* data()
    { return _x; }

  private:

    // Length of array
    std::size_t _size;

    // Array data
    T* _x;

  };

}

#endif
