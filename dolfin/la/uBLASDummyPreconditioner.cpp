// Copyright (C) 2006 Anders Logg
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
// First added:  2006-07-04
// Last changed: 2006-07-04

#include "uBLASVector.h"
#include "uBLASDummyPreconditioner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
uBLASDummyPreconditioner::uBLASDummyPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASDummyPreconditioner::~uBLASDummyPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBLASDummyPreconditioner::solve(uBLASVector& x, const uBLASVector& b) const
{
  x.vec().assign(b.vec());
}
//-----------------------------------------------------------------------------
