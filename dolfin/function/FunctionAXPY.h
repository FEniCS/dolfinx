// Copyright (C) 2013 Johan Hake
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
// First added:  2013-02-11
// Last changed: 2013-02-14

#ifndef __FUNCTION_AXPY_H
#define __FUNCTION_AXPY_H

#include <utility>
#include <vector>

namespace dolfin
{

  class Function;

  // FIXME: Straighten out memory management in this class by
  // switching to smart pointers.

  /// This class represents a linear combination of functions. It is
  /// mostly used as an intermediate class for operations such as u =
  /// 3*u0 + 4*u1; where the rhs generates an FunctionAXPY.
  class FunctionAXPY
  {

  public:

    /// Enum to decide what way AXPY is constructed
    enum class Direction : int {ADD_ADD=0, SUB_ADD=1, ADD_SUB=2, SUB_SUB=3};

    /// Constructor
    FunctionAXPY(const Function& func, double scalar);

    /// Constructor
    FunctionAXPY(const FunctionAXPY& axpy, double scalar);

    /// Constructor
    FunctionAXPY(const Function& func0, const Function& func1,
                 Direction direction);

    /// Constructor
    FunctionAXPY(const FunctionAXPY& axpy, const Function& func,
                 Direction direction);

    /// Constructor
    FunctionAXPY(const FunctionAXPY& axpy0, const FunctionAXPY& axpy1,
                 Direction direction);

    /// Constructor
    FunctionAXPY(std::vector<std::pair<double, const Function*> > pairs);

    /// Copy constructor
    FunctionAXPY(const FunctionAXPY& axpy);

    /// Destructor
    ~FunctionAXPY();

    /// Addition operator
    FunctionAXPY operator+(const Function& func) const;

    /// Addition operator
    FunctionAXPY operator+(const FunctionAXPY& axpy) const;

    /// Subtraction operator
    FunctionAXPY operator-(const Function& func) const;

    /// Subtraction operator
    FunctionAXPY operator-(const FunctionAXPY& axpy) const;

    /// Scale operator
    FunctionAXPY operator*(double scale) const;

    /// Scale operator
    FunctionAXPY operator/(double scale) const;

    /// Return the scalar and Function pairs
    const std::vector<std::pair<double, const Function*> >& pairs() const;

  private:

    /// Register another AXPY object
    void _register(const FunctionAXPY& axpy0, double scale);

    std::vector<std::pair<double, const Function*>> _pairs;

  };

}

#endif
