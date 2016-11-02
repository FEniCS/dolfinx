// This is a DOLFIN header file for predicates.cpp which provides
//
//   Routines for Arbitrary Precision Floating-point Arithmetic
//   and Fast Robust Geometric Predicates
//
// by
//
//   Jonathan Richard Shewchuk
//
// Code is placed in the public domain.

#ifndef PREDICATES_H
#define PREDICATES_H

#include <dolfin/log/log.h>

// Initialize tolerances for exact arithmetic
void exactinit();

// Compute relative orientation of points pa, pb, pc
double orient2d(double* pa, double* pb, double* pc);

// Compute relative orientation of points pa, pb, pc, pd
double orient3d(double* pa, double* pb, double* pc, double *pd);

namespace dolfin
{

  // Class used for automatic initialization of tolerances at startup.
  // A global instance is defined inside predicates.cpp to ensure that
  // the constructor and thus exactinit() is called.

  class PredicateInitialization
  {
  public:

    PredicateInitialization()
    {
      exactinit();
    }

  };

}

#endif
