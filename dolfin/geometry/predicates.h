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

namespace dolfin
{

  // Initialize tolerances for exact arithmetic
  void exactinit();

  // Compute relative orientation of points pa, pb, pc. The orientation
  // is such that orient2d(pa, pb, pc) > 0 if pa, pb, pc are ordered
  // counter-clockwise.
  double orient2d(double* pa, double* pb, double* pc);

  double orient2d(Point pa, Point pb, Point pc)
  {
    return orient2d(pa.coordinates(),
		    pb.coordinates(),
		    pc.coordinates());
  }

  // Compute relative orientation of points pa, pb, pc, pd. The
  // orientation is such that orient3d(pa, pb, pc, pd) > 0 if pa, pb,
  // pc, pd are oriented according to the left hand rule.
  double orient3d(double* pa, double* pb, double* pc, double* pd);

  double orient3d(Point pa, Point pb, Point pc, Point pd)
  {
    return orient3d(pa.coordinates(),
		    pb.coordinates(),
		    pc.coordinates(),
		    pd.coordinates());
  }

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
