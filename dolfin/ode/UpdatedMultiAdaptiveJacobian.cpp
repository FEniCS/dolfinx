// Copyright (C) 2005-2008 Anders Logg
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
// First added:  2005-01-27
// Last changed: 2008-04-22

#include <dolfin/common/constants.h>
#include <dolfin/la/uBLASVector.h>
#include "ODE.h"
#include "Method.h"
#include "MultiAdaptiveTimeSlab.h"
#include "MultiAdaptiveNewtonSolver.h"
#include "UpdatedMultiAdaptiveJacobian.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UpdatedMultiAdaptiveJacobian::UpdatedMultiAdaptiveJacobian
(MultiAdaptiveNewtonSolver& newton, MultiAdaptiveTimeSlab& timeslab)
  : TimeSlabJacobian(timeslab), newton(newton), ts(timeslab),  h(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
UpdatedMultiAdaptiveJacobian::~UpdatedMultiAdaptiveJacobian()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint UpdatedMultiAdaptiveJacobian::size(uint dim) const
{
  return ts.nj;
}
//-----------------------------------------------------------------------------
void UpdatedMultiAdaptiveJacobian::mult(const uBLASVector& x,
					uBLASVector& y) const
{
  // Compute product by the approximation y = J(u) x = (F(u + hx) - F(u)) / h.
  // Since Feval() compute -F rather than F, we compute according to
  //
  //     y = J(u) x = (-F(u - hx) - (-F(u))) / h

  // Update values, u <-- u - hx
  for (unsigned int j = 0; j < ts.nj; j++)
    ts.jx[j] -= h*x[j];

  // Compute -F(u - hx)
  newton.Feval(y);

  // Restore values, u <-- u + hx
  for (unsigned int j = 0; j < ts.nj; j++)
    ts.jx[j] += h*x[j];

  // Compute difference, using already computed -F(u)
  y -= newton. b;
  y /= h;
}
//-----------------------------------------------------------------------------
void UpdatedMultiAdaptiveJacobian::init()
{
  // Compute size of increment
  double umax = 0.0;
  for (unsigned int i = 0; i < ts.N; i++)
    umax = std::max(umax, std::abs(to_double(ts.u0[i]))); //Ok to convert to double here?
  h = std::max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * umax);
}
//-----------------------------------------------------------------------------
