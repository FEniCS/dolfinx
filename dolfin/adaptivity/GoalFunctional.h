// Copyright (C) 2010--2012 Marie E. Rognes
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
// First added:  2010-09-16
// Last changed: 2012-11-14

#ifndef __GOAL_FUNCTIONAL_H
#define __GOAL_FUNCTIONAL_H

#include <memory>
#include <dolfin/fem/Form.h>
#include "ErrorControl.h"

namespace dolfin
{

  /// A _GoalFunctional_ is a _Form_ of rank 0 with an associated
  /// _ErrorControl_.

  class GoalFunctional : public Form
  {

  public:

    // FIXME: The rank argument is unnecessary, a GoalFunction should
    // always have rank 0. The argument should be removed for that
    // reason.

    /// Create _GoalFunctional_
    ///
    /// *Arguments*
    ///     rank (int)
    ///         the rank of the functional (should be 0)
    ///     num_coefficients (int)
    ///         the number of coefficients in functional
    GoalFunctional(std::size_t rank, std::size_t num_coefficients);

    /// Update error control instance with given forms
    ///
    /// *Arguments*
    ///     a (_Form_)
    ///         a bilinear form
    ///     L (_Form_)
    ///         a linear form
    virtual void update_ec(const Form& a, const Form& L) = 0;

    // Pointer to _ErrorControl_ instance
    std::shared_ptr<ErrorControl> _ec;

  };

}
#endif
