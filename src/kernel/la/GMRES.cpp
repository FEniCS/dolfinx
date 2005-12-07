// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-12-02
// Last changed:

#include <dolfin/GMRES.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GMRES::GMRES() : KrylovSolver(gmres)
{
  // Do nothing.
}
//-----------------------------------------------------------------------------
GMRES::GMRES(Preconditioner::Type preconditioner_type) : KrylovSolver(gmres, preconditioner_type)
{
  // Do nothing.
}
//-----------------------------------------------------------------------------
GMRES::~GMRES()
{
  // Do nothing.
}
//-----------------------------------------------------------------------------
