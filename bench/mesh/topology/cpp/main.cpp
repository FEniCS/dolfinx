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
// First added:  2010-11-25
// Last changed: 2012-12-12

#include <dolfin.h>
#include <dolfin/log/LogLevel.h>

using namespace dolfin;

#define NUM_REPS 5
#define SIZE 64

// Use for quick testing
//#define NUM_REPS 2
//#define SIZE 32

int main(int argc, char* argv[])
{
  info("Creating cell-cell connectivity for unit cube of size %d x %d x %d (%d repetitions)",
       SIZE, SIZE, SIZE, NUM_REPS);

  set_log_level(DBG);

  parameters.parse(argc, argv);

  UnitCubeMesh mesh(SIZE, SIZE, SIZE);
  const int D = mesh.topology().dim();

  // Clear timing (if there is some)
  { Timer t("Compute connectivity 3-3"); }
  timing("Compute connectivity 3-3", TimingClear::clear);

  for (int i = 0; i < NUM_REPS; i++)
  {
    mesh.clean();
    mesh.init(D, D);
    dolfin::cout << "Created unit cube: " << mesh << dolfin::endl;
  }

  // Report timings
  list_timings(TimingClear::keep,
               { TimingType::wall, TimingType::user, TimingType::system });

  // Report timing
  const auto t = timing("Compute connectivity 3-3", TimingClear::clear);
  info("BENCH %g", std::get<1>(t));

  return 0;
}
