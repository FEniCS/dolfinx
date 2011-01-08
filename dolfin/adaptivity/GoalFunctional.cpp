// Copyright (C) 2010 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
//
// Modified by Anders Logg, 2010.
//
// First added:  2010-09-16
// Last changed: 2010-12-02

#include "GoalFunctional.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GoalFunctional::GoalFunctional(uint rank, uint num_coefficients)
  : Form(rank, num_coefficients), _ec(0)
{
  // Check that rank is 0
  assert(rank == 0);
}
//-----------------------------------------------------------------------------
// boost::scoped_ptr<ErrorControl> GoalFunctional::ec()
// {
//   assert(_ec);
//   return _ec;
// }
//-----------------------------------------------------------------------------
