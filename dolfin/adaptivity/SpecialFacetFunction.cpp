// Copyright (C) 2010 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
//
// First added:  2010-10-13
// Last changed: 2010-11-05

#include <dolfin/common/Array.h>
#include <dolfin/function/Data.h>
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
void SpecialFacetFunction::eval(Array<double>& values, const Data& data) const
{
  values[0] = 0.0;
  if (data.on_facet())
    _f_e[data.facet()]->eval(values, data);
}
//-----------------------------------------------------------------------------
