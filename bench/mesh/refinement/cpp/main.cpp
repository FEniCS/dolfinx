// Copyright (C) 2006-2010 Anders Logg
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
// First added:  2006-11-01
// Last changed: 2012-12-12

#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 5
#define SIZE 4

int main(int argc, char* argv[])
{
  info("Uniform refinement of unit cube of size %d x %d x %d (%d refinements)",
       SIZE, SIZE, SIZE, NUM_REPS);

  parameters.parse(argc, argv);

  UnitCubeMesh unitcube_mesh(SIZE, SIZE, SIZE);
  Mesh mesh(unitcube_mesh);

  tic();
  for (int i = 0; i < NUM_REPS; i++)
  {
    mesh = refine(mesh);
    dolfin::cout << "Refined mesh: " << mesh << dolfin::endl;
  }
  info("BENCH %g", toc());

  return 0;
}
