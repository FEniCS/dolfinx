// Copyright (C) 2006 Garth N. Wells
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
// Modified by Anders Logg, 2010.
//
// First added:  2006-08-18
// Last changed: 2010-05-03

#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 100
#define SIZE 10000000

int main(int argc, char* argv[])
{
  info("Assigning to vector of size %d (%d repetitions)",
       SIZE, NUM_REPS);

  parameters.parse(argc, argv);

  Vector x(MPI_COMM_WORLD, SIZE);

  for (unsigned int i = 0; i < NUM_REPS; i++)
    for (unsigned int j = 0; j < SIZE; j++)
      x.setitem(j, 1.0);

  return 0;
}
