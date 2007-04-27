// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007
//
// First added:  2006-02-21
// Last changed: 2007-04-27

#include <dolfin/dolfin_log.h>
#include <dolfin/Form.h>
#include <dolfin/GenericPDE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericPDE::GenericPDE(Form& a,
                       Form& L,
                       Mesh& mesh,
                       Array<BoundaryCondition*>& bcs)
  : a(a), L(L), mesh(mesh), bcs(bcs)
{
  // Check ranks of forms
  if ( a.form().rank() != 2 )
    dolfin_error1("Expected a bilinear form but rank is %d.", a.form().rank());
  if ( L.form().rank() != 1 )
    dolfin_error1("Expected a linear form but rank is %d.", L.form().rank());
}
//-----------------------------------------------------------------------------
GenericPDE::~GenericPDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
