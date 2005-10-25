// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-24
// Last changed: 2005

#include <dolfin/NonlinearFunctional.h>

using namespace dolfin;
NonlinearFunctional::NonlinearFunctional() : mesh(0), a(0), L(0), b(0), x(0)
{
//FIXME
}
//-----------------------------------------------------------------------------
NonlinearFunctional::~NonlinearFunctional()
{
// Do nothing 
}
//-----------------------------------------------------------------------------
