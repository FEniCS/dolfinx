// Copyright (C) 2009-2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "GenericFunction.h"
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/log/log.h>
#include <string>

using namespace dolfin;
using namespace dolfin::function;

//-----------------------------------------------------------------------------
GenericFunction::GenericFunction() : common::Variable("u", "a function")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericFunction::~GenericFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GenericFunction::eval(Eigen::Ref<EigenRowArrayXXd> values,
                           Eigen::Ref<const EigenRowArrayXXd> x,
                           const mesh::Cell& cell) const
{
  // Redirect to simple eval
  eval(values, x);
}
//-----------------------------------------------------------------------------
void GenericFunction::eval(Eigen::Ref<EigenRowArrayXXd> values,
                           Eigen::Ref<const EigenRowArrayXXd> x) const
{
  log::dolfin_error("GenericFunction.cpp", "evaluate function (Eigen version)",
                    "Missing eval() function (must be overloaded)");
}
//-----------------------------------------------------------------------------
std::size_t GenericFunction::value_size() const
{
  std::size_t size = 1;
  for (std::size_t i = 0; i < value_rank(); ++i)
    size *= value_dimension(i);
  return size;
}
//-----------------------------------------------------------------------------
