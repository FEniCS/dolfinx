// Copyright (C) 2010 Marie E. Rognes
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
// First added:  2010-09-16
// Last changed: 2010-12-02

#include "GoalFunctional.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GoalFunctional::GoalFunctional(std::size_t rank, std::size_t num_coefficients)
  : Form(rank, num_coefficients)
{
  // Check that rank is 0
  dolfin_assert(rank == 0);
}
//-----------------------------------------------------------------------------
