// Copyright (C) 2006-2009 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Martin Sandve Alnes, 2008.
// Modified by Garth N. Wells, 2008-2011.
//
// First added:  2006-02-09
// Last changed: 2011-05-15

#ifndef __CONSTANT_H
#define __CONSTANT_H

#include <vector>
#include "Expression.h"

namespace dolfin
{

  class Mesh;

  /// This class represents a constant-valued expression.

  class Constant : public Expression
  {
  public:

    /// Create scalar constant
    Constant(double value);

    /// Create vector constant (dim = 2)
    Constant(double value0, double value1);

    /// Create vector constant (dim = 3)
    Constant(double value0, double value1, double value2);

    /// Create vector-valued constant
    Constant(std::vector<double> values);

    /// Create tensor-valued constant for flattened array of values
    Constant(std::vector<uint> value_shape,
             std::vector<double> values);

    /// Copy constructor
    Constant(const Constant& constant);

    /// Destructor
    ~Constant();

    /// Assignment operator
    const Constant& operator= (const Constant& constant);

    /// Assignment operator
    const Constant& operator= (double constant);

    /// Cast to double (for scalar constants)
    operator double() const;

    //--- Implementation of Expression interface ---

    void eval(Array<double>& values, const Array<double>& x) const;

  private:

    // Values of constant function
    std::vector<double> _values;

  };

}

#endif
