// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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
