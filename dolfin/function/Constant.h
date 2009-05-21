// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2006-02-09
// Last changed: 2008-11-17

#ifndef __CONSTANT_H
#define __CONSTANT_H

#include <vector>
#include "Function.h"

namespace dolfin
{

  /// This class implements the functionality for functions that take
  /// a single constant value.

  class Constant : public Function
  {
  public:

    /// Create constant scalar function with given value
    explicit Constant(double value);

    /// Create constant vector function with given size and value
    Constant(uint size, double value);

    /// Create constant vector function with given size and values
    Constant(const std::vector<double>& values);

    /// Create constant tensor function with given shape and values
    Constant(const std::vector<uint>& shape, const std::vector<double>& values);

    /// Copy constructor
    Constant(const Constant& c);

    /// Assignment operator
    const Constant& operator= (const Constant& c);

    /// Destructor
    ~Constant();

    /// Function evaluation
    void eval(double* values, const Data& data) const;

    /// Return size
    uint size() const
    { return _size; }

    /// Return values
    const double* values() const 
    { return _values; }

  private:

    // Size of value (number of entries in tensor value)
    uint _size;

    // Values of constant function
    double* _values;

  };

}

#endif
