// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/LinearForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearForm::LinearForm()
{
  
}
//-----------------------------------------------------------------------------
LinearForm::~LinearForm()
{

}
//-----------------------------------------------------------------------------
real LinearForm::operator() (TestFunction v)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
