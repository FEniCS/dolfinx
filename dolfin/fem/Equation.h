// Copyright (C) 2011 Anders Logg
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
// First added:  2011-06-21
// Last changed: 2011-06-22

#ifndef __EQUATION_H
#define __EQUATION_H

#include <memory>

namespace dolfin
{

  // Forward declarations
  class Form;

  /// This class represents a variational equation lhs == rhs.
  /// The equation can be either linear or nonlinear:
  ///
  /// 1. Linear (a == L), in which case a must be a bilinear form
  ///    and L must be a linear form.
  ///
  /// 2. Nonlinear (F == 0), in which case F must be a linear form.

  class Equation
  {
  public:

    /// Create equation a == L
    /// @param[in] a (Form)
    ///  Form representing the LHS
    /// @param[in] L (Form)
    ///  Form representing the RHS
    Equation(std::shared_ptr<const Form> a,
             std::shared_ptr<const Form> L);

    /// Create equation F == 0
    /// @param[in] F (Form)
    /// @param[in] rhs (int)
    Equation(std::shared_ptr<const Form> F, int rhs);

    /// Destructor
    ~Equation();

    /// Check whether equation is linear
    /// @return bool
    bool is_linear() const;

    /// Return form for left-hand side
    /// @return Form
    /// LHS form
    std::shared_ptr<const Form> lhs() const;

    /// Return form for right-hand side
    /// @return Form
    /// RHS form
    std::shared_ptr<const Form> rhs() const;

    /// Return value for right-hand side
    /// @return int
    int rhs_int() const;

  private:

    // Left-hand side form
    std::shared_ptr<const Form> _lhs;

    // Right-hand side form
    std::shared_ptr<const Form> _rhs;

    // Right-hand side value (should be zero)
    int _rhs_int;

    // Flag for whether equation is linear
    bool _is_linear;

  };

}

#endif
