// Copyright (C) 2010 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
//
// First added:  2010-10-13
// Last changed: 2011-01-04

#include <dolfin/common/Array.h>
#include "SpecialFacetFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SpecialFacetFunction::SpecialFacetFunction(std::vector<Function*> f_e, uint dim)
  : Expression(dim)
{
  for (uint i=0; i < f_e.size(); i++)
    _f_e.push_back(f_e[i]);
}
//-----------------------------------------------------------------------------
SpecialFacetFunction::SpecialFacetFunction(std::vector<Function*> f_e)
  : Expression()
{
  for (uint i=0; i < f_e.size(); i++)
    _f_e.push_back(f_e[i]);
}
//-----------------------------------------------------------------------------
Function* SpecialFacetFunction::operator[] (uint i) const {
  return _f_e[i];
}
//-----------------------------------------------------------------------------
void SpecialFacetFunction::eval(Array<double>& values, const Array<double>& x,
                                const ufc::cell& cell) const
{
  values[0] = 0.0;
  if (cell.local_facet >= 0)
    _f_e[cell.local_facet]->eval(values, x, cell);
}
//-----------------------------------------------------------------------------
