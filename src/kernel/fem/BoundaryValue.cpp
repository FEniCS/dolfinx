// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006
//
// First added:  2005-02-13
// Last changed: 2006-04-21

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
const BoundaryValue& BoundaryValue::operator= (real value)
{
  set(value);
  return *this;
}
//-----------------------------------------------------------------------------
void BoundaryValue::set(real value)
{
  fixed = true;
  this->value = value;
}
//-----------------------------------------------------------------------------
void BoundaryValue::reset()
{
  fixed = false;
  value = 0.0;
}
//-----------------------------------------------------------------------------
