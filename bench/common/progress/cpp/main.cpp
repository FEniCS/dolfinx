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
// First added:  2010-06-29
// Last changed: 2010-11-16

#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 5
#define SIZE 500000000

int main(int argc, char* argv[])
{
  info("Creating progress bar with %d steps (%d repetitions)",
       SIZE, NUM_REPS);

  for (int i = 0; i < NUM_REPS; i++)
  {
    Progress p("Stepping", SIZE);
    double sum = 0.0;
    for (int j = 0; j < SIZE; j++)
    {
      sum += 0.1;
      p++;
    }
    dolfin::cout << "sum = " << sum << dolfin::endl;
  }

  return 0;
}
