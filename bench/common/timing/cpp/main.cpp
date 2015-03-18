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
// First added:  2010-11-16
// Last changed: 2010-11-16

#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 10000000

int main(int argc, char* argv[])
{
  info("Timing access and registration of timings (%d repetitions)", NUM_REPS);

  // Access to timer
  double sum = 0.0;
  double t0 = time();
  double t1 = t0;
  Timer timer_function("time() function");
  for (int i = 0; i < NUM_REPS; i++)
  {
    t0 = time();
    t1 = time();
    sum += t1 - t0;
  }
  timer_function.stop();
  dolfin::cout << "sum = " << sum << dolfin::endl << dolfin::endl;

  // Test timer
  Timer timer_loop("timer start/stop");
  Timer timer_class("Timer class");
  for (int i = 0; i < NUM_REPS; i++)
  {
    timer_loop.start();
    timer_loop.stop();
  }
  timer_class.stop();

  // Report timings
  list_timings(TimingClear::keep,
               { TimingType::wall, TimingType::user, TimingType::system });

  return 0;
}
