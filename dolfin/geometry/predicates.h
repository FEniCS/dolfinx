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

#ifndef __PREDICATES_H
#define __PREDICATES_H

namespace dolfin
{

  class Point;

  // Initialize tolerances for exact arithmetic
  void exactinit();

  // Compute relative orientation of points a, b, c. The orientation
  // is such that orient2d(a, b, c) > 0 if a, b, c are ordered
  // counter-clockwise.
  double _orient2d(const double* a, const double* b, const double* c);

  // Convenience function using dolfin::Point
  double orient2d(const Point& a, const Point& b, const Point& c);

  // Compute relative orientation of points a, b, c, d. The orientation
  // is such that orient3d(a, b, c, d) > 0 if a, b, c, d are oriented
  // according to the left hand rule.
  double _orient3d(const double* a, const double* b, const double* c, const double* d);

  // Convenience function using dolfin::Point
  double orient3d(const Point& a, const Point& b, const Point& c, const Point& d);

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
