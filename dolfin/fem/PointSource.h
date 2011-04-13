// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-04-13
// Last changed: 2011-04-13

#include <boost/shared_ptr.hpp>
#include <dolfin/mesh/Point.h>

namespace dolfin
{

  // Forward declarations
  class FunctionSpace;
  class GenericVector;

  /// This class provides an easy mechanism for adding a point source
  /// (Dirac delta function) to the right-hand side vector in a
  /// variational problem. The associated function space must be
  /// scalar in order for the inner product with the (scalar) Dirac
  /// delta function to be well defined.

  class PointSource
  {
  public:

    /// Create point source at given point of given magnitude
    PointSource(const FunctionSpace& V,
                const Point& p,
                double magnitude=1.0);

    /// Create point source at given point of given magnitude
    PointSource(boost::shared_ptr<const FunctionSpace> V,
                const Point& p,
                double magnitude=1.0);

    /// Destructor
    ~PointSource();

    /// Apply (add) point source to right-hand side vector
    void apply(GenericVector& b);

  private:

    // Check that function space is scalar
    void check_is_scalar(const FunctionSpace& V);

    // The function space
    boost::shared_ptr<const FunctionSpace> V;

    // The point
    Point p;

    // Magnitude
    double magnitude;

  };

}
