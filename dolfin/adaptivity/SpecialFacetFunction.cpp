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
// First added:  2010-10-13
// Last changed: 2011-01-04

#include <dolfin/common/Array.h>
#include <dolfin/function/Function.h>
#include "SpecialFacetFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SpecialFacetFunction::SpecialFacetFunction(std::vector<Function>& f_e, uint dim)
  : Expression(dim), f_e(f_e)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SpecialFacetFunction::SpecialFacetFunction(std::vector<Function>& f_e)
  : Expression(), f_e(f_e)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function& SpecialFacetFunction::operator[] (uint i) const
{
  assert(i < f_e.size());
  return f_e[i];
}
//-----------------------------------------------------------------------------
void SpecialFacetFunction::eval(Array<double>& values, const Array<double>& x,
                                const ufc::cell& cell) const
{
  values[0] = 0.0;
  if (cell.local_facet >= 0)
    f_e[cell.local_facet].eval(values, x, cell);
}
//-----------------------------------------------------------------------------
