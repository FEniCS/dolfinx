// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/BilinearForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BilinearForm::BilinearForm()
{
  
}
//-----------------------------------------------------------------------------
BilinearForm::~BilinearForm()
{

}
//-----------------------------------------------------------------------------
real BilinearForm::operator() (TrialFunction u, TestFunction v)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
