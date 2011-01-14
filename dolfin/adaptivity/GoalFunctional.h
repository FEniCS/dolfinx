// Copyright (C) 2010 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
//
// First added:  2010-09-16
// Last changed: 2010-12-02

#ifndef __GOAL_FUNCTIONAL_H
#define __GOAL_FUNCTIONAL_H

#include <boost/scoped_ptr.hpp>
#include <dolfin/fem/Form.h>
#include "ErrorControl.h"

namespace dolfin
{

  /// A _GoalFunctional_ is a _Form_ of rank 0 with an associated
  /// _ErrorControl_.

  class GoalFunctional : public Form
  {

  public:

    // FIXME: Remove rank argument

    /// Create _GoalFunctional_
    ///
    /// *Arguments*
    ///     rank (int)
    ///         the rank of the functional (should be 0)
    ///
    ///     num_coefficients (int)
    ///         the number of coefficients in functional
    GoalFunctional(uint rank, uint num_coefficients);

    /// Update error control instance with given forms
    ///
    /// *Arguments*
    ///     a (_Form_)
    ///         a bilinear form
    ///     L (_Form_)
    ///         a linear form
    virtual void update_ec(const Form& a, const Form& L) = 0;

    /// Pointer to _ErrorControl_ instance
    boost::scoped_ptr<ErrorControl> _ec;

  };

}
#endif
