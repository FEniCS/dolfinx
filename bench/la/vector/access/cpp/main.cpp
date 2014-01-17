// Copyright (C) 2010 Anders Logg
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
// First added:  2010-03-30
// Last changed: 2010-05-03

#include <dolfin.h>

using namespace dolfin;

#define SIZE 10000000
#define NUM_REPS 100

int main(int argc, char* argv[])
{
  info("Accessing vector of size %d (%d repetitions)",
       SIZE, NUM_REPS);

  parameters.parse(argc, argv);

  Vector x(MPI_COMM_WORLD, SIZE);
  x.zero();

  double sum = 0.0;
  for (unsigned int i = 0; i < NUM_REPS; i++)
    for (unsigned int j = 0; j < SIZE; j++)
      sum += x[j];
  dolfin::cout << "Sum is " << sum << dolfin::endl;

  return 0;
}
