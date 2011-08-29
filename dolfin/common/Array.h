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
// Modified by Anders Logg, 2010.
//
// First added:  2009-12-06
// Last changed: 2011-05-17

#ifndef __DOLFIN_ARRAY_H
#define __DOLFIN_ARRAY_H

#include <sstream>
#include <string>
#include <utility>
#include <boost/shared_array.hpp>

#include <dolfin/common/constants.h>
#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>

#include "NoDeleter.h"


namespace dolfin
{

  /// This class provides a simple wrapper for a pointer to an array. A purpose
  /// of this class is to enable the simple and safe exchange of data between
  /// C++ and Python.

  template <class T> class Array
  {
  public:

    /// Create empty array
    Array() : _size(0), x(0) {}

    /// Create array of size N
    explicit Array(uint N) : _size(N), x(new T[N]) {}

    /// Copy constructor (arg name need to have a different name that 'x')
    Array(const Array& other) : _size(0), x(0)
    { *this = other; }

    /// Construct array from a shared pointer
    Array(uint N, boost::shared_array<T> x) : _size(N), x(x) {}

    /// Construct array from a pointer. Array will not take ownership.
    Array(uint N, T* x) : _size(N), x(boost::shared_array<T>(x, NoDeleter())) {}

    /// Assignment operator
    const Array& operator= (const Array& x)
    {
      // Resize if necessary
      if (x.size() == 0 && !x.x)
      {
        this->x.reset();
        this->_size = 0;
      }
      else if (this->_size != x.size())
      {
        this->x.reset(new T[x.size()]);
        this->_size = x.size();
      }

      // Copy data
      if (_size > 0)
      {
        assert(this->x);
        assert(x.x);
        std::copy(&x.x[0], &x.x[_size], &this->x[0]);
      }

      return *this;
    }

    /// Construct array from a pointer. Array will not take ownership.
    void update(uint N, T* _x)
    {
      _size = N;
      x.reset(_x, NoDeleter());
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

    /// Resize array to size N. If size changes, contents will be destroyed.
    void resize(uint N)
    {
      if (N == _size)
        return;
      else
      {
        // FIXME: Do we want to allow resizing of shared data?
        if (x.unique())
        {
          _size = N;
          x.reset(new T[N]);
        }
        else
          dolfin_error("Array.h",
                       "resize Array",
                       "Data is shared");
      }
    }

    /// Return size of array
    uint size() const
    { return _size; }

    /// Zero array
    void zero()
    { assert(x); std::fill(&x[0], &x[_size], 0.0); }

    /// Set entries which meet (abs(x[i]) < eps) to zero
    void zero_eps(double eps=DOLFIN_EPS);

    /// Return minimum value of array
    T min() const
    { assert(x); return *std::min_element(&x[0], &x[_size]); }

    /// Return maximum value of array
    T max() const
    { assert(x); return *std::max_element(&x[0], &x[_size]); }

    /// Access value of given entry (const version)
    const T& operator[] (uint i) const
    { assert(x); assert(i < _size); return x[i]; }

    /// Access value of given entry (non-const version)
    T& operator[] (uint i)
    {
      assert(x);
      assert(i < _size);
      return x[i];
    }

    /// Assign value to all entries
    const Array<T>& operator= (T& x)
    {
      assert(this->x);
      for (uint i = 0; i < _size; ++i)
        this->x[i] = x;
      return *this;
    }

    /// Return pointer to data (const version)
    const boost::shared_array<T> data() const
    { return x; }

    /// Return pointer to data (non-const version)
    boost::shared_array<T> data()
    { return x; }

  private:

    // Length of array
    dolfin::uint _size;

    // Array data
    boost::shared_array<T> x;

  };

  template <class T>
  inline void Array<T>::zero_eps(double eps)
  {
    error("Array<T>::zero_eps can only be used for T=double.");
  }

  template <>
  inline void Array<double>::zero_eps(double eps)
  {
    assert(x);
    for (uint i = 0; i < _size; ++i)
    {
      if (std::abs(x[i]) < eps)
        x[i] = 0.0;
    }
  }

}

#endif
