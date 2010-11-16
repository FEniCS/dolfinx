// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
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

  summary();

  return 0;
}
