
#pragma once

#define REAL double /* float or double */

namespace dolfin
{
namespace geometry
{
class Point;

/// Initialize tolerances for exact arithmetic
void exactinit();

/// Compute relative orientation of point x wrt segment [a, b]
double orient1d(double a, double b, double x);

/// Compute relative orientation of points a, b, c. The orientation
/// is such that orient2d(a, b, c) > 0 if a, b, c are ordered
/// counter-clockwise.
REAL _orient2d(const REAL* a, const REAL* b, const REAL* c);

/// Convenience function using dolfin::Point
double orient2d(const Point& a, const Point& b, const Point& c);

/// Compute relative orientation of points a, b, c, d. The
/// orientation is such that orient3d(a, b, c, d) > 0 if a, b, c, d
/// are oriented according to the left hand rule.
REAL _orient3d(const REAL* a, const REAL* b, const REAL* c, const REAL* d);

/// Convenience function using dolfin::Point
double orient3d(const Point& a, const Point& b, const Point& c, const Point& d);

/// Class used for automatic initialization of tolerances at startup.
/// A global instance is defined inside predicates.cpp to ensure that
/// the constructor and thus exactinit() is called.

class PredicateInitialization
{
public:
  PredicateInitialization() { exactinit(); }
};
} // namespace geometry
} // namespace dolfin
