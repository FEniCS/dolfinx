// Copyright (C) 2010 Marie E. Rognes
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstring>

#include "SpecialFacetFunction.h"
#include <dolfin/function/Function.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SpecialFacetFunction::SpecialFacetFunction(std::vector<Function>& f_e)
    : Expression({}), _f_e(f_e)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SpecialFacetFunction::SpecialFacetFunction(std::vector<Function>& f_e,
                                           std::size_t dim)
    : Expression({dim}), _f_e(f_e)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SpecialFacetFunction::SpecialFacetFunction(std::vector<Function>& f_e,
                                           std::vector<std::size_t> value_shape)
    : Expression(value_shape), _f_e(f_e)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function& SpecialFacetFunction::operator[](std::size_t i) const
{
  dolfin_assert(i < _f_e.size());
  return _f_e[i];
}
//-----------------------------------------------------------------------------
void SpecialFacetFunction::eval(Eigen::Ref<Eigen::VectorXd> values,
                                Eigen::Ref<const Eigen::VectorXd> x,
                                const ufc::cell& cell) const
{
  memset(values.data(), 0, values.size() * sizeof(double));
  if (cell.local_facet >= 0)
    _f_e[cell.local_facet].eval(values, x, cell);
}
//-----------------------------------------------------------------------------
