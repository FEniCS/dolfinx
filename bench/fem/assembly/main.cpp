// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-11-19
// Last changed: 2007-05-24

#include <dolfin.h>
#include "Advection.h"

const unsigned int num_repetitions = 10;

using namespace dolfin;

int main()
{
  UnitCube mesh(8, 8, 8);
  Function w(mesh, 1.0);
  AdvectionBilinearForm a(w);
  Matrix A;
  
  tic();
  for (unsigned int i = 0; i < num_repetitions; i++)
    assemble(A, a, mesh);
  double t = toc();
  t /= static_cast<double>(num_repetitions);

  message("Time to assemble matrix: %.3g", t);

  return 0;
}
