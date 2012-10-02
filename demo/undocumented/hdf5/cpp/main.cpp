// Copyright (C) 2012 Chris Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2012-07-16
// Last changed:
//
// This demo program illustrates the use of (parallel)HDF5 for
// scalable file IO

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

int main()
{
  // Create mesh and function space
  UnitSquare mesh(128, 128);
  Poisson::FunctionSpace V(mesh);

  // Create function
  Function u(V);

  // Write function to XDMF file
  File file("u.xdmf");
  file << u;

  return 0;
}
