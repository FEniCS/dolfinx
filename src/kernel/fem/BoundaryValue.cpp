// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-02-13
// Last changed: 2005

#include <dolfin/dolfin_log.h>
#include <dolfin/BoundaryValue.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryValue::BoundaryValue() : fixed(false), value(0.0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryValue::~BoundaryValue()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BoundaryValue::set(real value)
{
  fixed = true;
  this->value = value;
}
//-----------------------------------------------------------------------------
