// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2005-12-02
// Last changed: 2006-03-14

#include <dolfin/GMRES.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GMRES::GMRES() : KrylovSolver(gmres)
{
  // Do nothing.
}
//-----------------------------------------------------------------------------
GMRES::GMRES(Preconditioner::Type preconditioner) : KrylovSolver(gmres, preconditioner)
{
  // Do nothing.
}
//-----------------------------------------------------------------------------
GMRES::GMRES(Preconditioner& preconditioner) : KrylovSolver(gmres, preconditioner)
{
  // Do nothing.
}
//-----------------------------------------------------------------------------
GMRES::~GMRES()
{
  // Do nothing.
}
//-----------------------------------------------------------------------------
