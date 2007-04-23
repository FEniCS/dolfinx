// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-09-02
// Last changed: 2005

#include <dolfin/dolfin_log.h>
#include <dolfin/TimeDependent.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeDependent::TimeDependent() : t(0)
{
 // Do nothing
}
//-----------------------------------------------------------------------------
TimeDependent::TimeDependent(const real& t) : t(&t)
{
 // Do nothing
}
//-----------------------------------------------------------------------------
TimeDependent::~TimeDependent() 
{
 // Do nothing
}
//-----------------------------------------------------------------------------
void TimeDependent::sync(const real& t)
{
  this->t  = &t;
}
//-----------------------------------------------------------------------------
real TimeDependent::time() const
{
	if( !t )
    dolfin_error("Time has not been associated with object.");
		
	return *t;
}
//-----------------------------------------------------------------------------
