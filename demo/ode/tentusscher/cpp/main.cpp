// Copyright (C) Glenn Terje Lines, Ola Skavhaug and Simula Research Laboratory.
// Licensed under the GNU LGPL Version 2.1.
//
// Original code copied from PyCC.
// Modified by Anders Logg 2006.
//
// First added:  2006-05-24
// Last changed: 2008-10-07
//
// This demo solves the Courtemanche model for cardiac excitation.

#include <dolfin.h>
#include "tentusscher.h"

using namespace dolfin;

int main()
{
  dolfin_set("ODE tolerance", 1.0e-5);
  dolfin_set("ODE maximum time step", 100.0);
  dolfin_set("ODE nonlinear solver", "newton");
  dolfin_set("ODE linear solver", "iterative");
  dolfin_set("ODE initial time step", 0.25);

  tentusscher ode(500);
  ode.solve();

  return 0;
}
