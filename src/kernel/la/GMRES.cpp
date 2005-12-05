// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-12-02
// Last changed:

#include <dolfin/GMRES.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GMRES::GMRES() : KrylovSolver(KSPGMRES)
{
  // Do nothing.
}
//-----------------------------------------------------------------------------
GMRES::GMRES(PCType pctype) : KrylovSolver(KSPGMRES, pctype)
{
  // Do nothing.
}
//-----------------------------------------------------------------------------
GMRES::~GMRES()
{
  // Do nothing.
}
//-----------------------------------------------------------------------------
