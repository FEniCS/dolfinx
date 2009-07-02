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
  Tentusscher ode(500);
  ode.parameters("tolerance") = 1.0e-5;
  ode.parameters("maximum_time_step") = 100.0;
  ode.parameters("nonlinear_solver") = "newton";
  ode.parameters("linear_solver") = "iterative";
  ode.parameters("initial_time_step") = 0.25;

  ode.solve();

  return 0;
}
