// Copyright (C) 2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-06
// Last changed:

#ifndef __ARRAY_H
#define __ARRAY_H

#include <boost/shared_array.hpp>

#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>

namespace dolfin
{

  /// This class provides a simple vector-type class for doubles. A purpose of 
  /// this class is to enable the simple and safe exchange of data between C++
  /// and Python.

  class Array
  {
  public:

    /// Create array of size N
    explicit Array(uint N);

    /// Copy constructor
    explicit Array(const Array& x);

    /// Construct array from a shared pointer
    Array(uint N, boost::shared_array<double> x);

    /// Construct array from a pointer. Array will not take ownership.
    Array(uint N, double* x);

    /// Destructor
    ~Array();

    /// Assignment operator
    const Array& operator= (const Array& x);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Resize array to size N. If size changes, contents will be destroyed.
    void resize(uint N);

    /// Return size of array
    uint size() const;

    /// Zero array
    void zero();

    /// Return minimum value of array
    double min() const;

    /// Return maximum value of array
    double max() const;

    /// Access value of given entry (const version)
    double operator[] (uint i) const
    { assert(i < _size); return x[i]; };

    /// Access value of given entry (non-const version)
    double& operator[] (uint i)
    { assert(i < _size); return x[i]; };

    /// Return pointer to data (const version)
    const boost::shared_array<double> data() const
    { return x; }

    /// Return pointer to data (non-const version)
    boost::shared_array<double> data()
    { return x; }

  private:

    // Length of array
    dolfin::uint _size;

    // Array data
    boost::shared_array<double> x;

  };

}

#endif
