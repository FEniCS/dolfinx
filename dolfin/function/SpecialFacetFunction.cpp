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

#include <cstring>

#include <dolfin/common/Array.h>
#include <dolfin/function/Function.h>
#include "SpecialFacetFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SpecialFacetFunction::SpecialFacetFunction(std::vector<Function>& f_e,
                                           std::vector<std::size_t> value_shape)
  : Expression(value_shape), _f_e(f_e)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SpecialFacetFunction::SpecialFacetFunction(std::vector<Function>& f_e, std::size_t dim)
  : Expression(dim), _f_e(f_e)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SpecialFacetFunction::SpecialFacetFunction(std::vector<Function>& f_e)
  : Expression(), _f_e(f_e)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function& SpecialFacetFunction::operator[] (std::size_t i) const
{
  dolfin_assert(i < _f_e.size());
  return _f_e[i];
}
//-----------------------------------------------------------------------------
void SpecialFacetFunction::eval(Array<double>& values, const Array<double>& x,
                                const ufc::cell& cell) const
{
  memset(values.data(), 0, values.size()*sizeof(double));
  if (cell.local_facet >= 0)
    _f_e[cell.local_facet].eval(values, x, cell);
}
//-----------------------------------------------------------------------------
