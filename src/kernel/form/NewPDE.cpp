// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewPDE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewPDE::NewPDE()
{

}
//-----------------------------------------------------------------------------
NewPDE::~NewPDE()
{

}
//-----------------------------------------------------------------------------
real NewPDE::lhs(Form::TrialFunction u, Form::TestFunction v)
{
  return a(u,v);
}
//-----------------------------------------------------------------------------
real NewPDE::rhs(Form::TestFunction v)
{
  return l(v);
}
//-----------------------------------------------------------------------------
