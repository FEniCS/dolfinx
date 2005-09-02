// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-09-02
// Last changed: 2005

#include <dolfin/Synchronizer.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Synchronizer::Synchronizer() : time_set(false)
{
 // Do nothing
}
//-----------------------------------------------------------------------------
Synchronizer::Synchronizer(const real& t) : t(&t), time_set(true)
{
 // Do nothing
}
//-----------------------------------------------------------------------------
Synchronizer::~Synchronizer() 
{
 // Do nothing
}
//-----------------------------------------------------------------------------
void Synchronizer::sync(const real& t)
{
  this->t  = &t;
  time_set = true;
}
//-----------------------------------------------------------------------------
real Synchronizer::time() const
{
	if( !time_set)
    dolfin_error("Time has not been associated with object.");

	return *t;
}
//-----------------------------------------------------------------------------
