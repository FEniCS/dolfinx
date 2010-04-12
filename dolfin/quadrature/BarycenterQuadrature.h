// Copyright (C) 2010-03-17 André Massing.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010.
//
// First added:  2010-03-17
// Last changed: 2010-04-12

#ifndef  __BARYCENTER_QUADRATURE_H
#define  __BARYCENTER_QUADRATURE_H

#ifdef HAS_CGAL

#include "BarycenterQuadrature.h"

#include <vector>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <dolfin/mesh/Point.h>

namespace dolfin
{

  typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
  typedef CGAL::Nef_polyhedron_3<Kernel> Nef_polyhedron_3;

  /// This class computes the barycenter of an arbitrary polyhedron or
  /// polygon in 3D and therefore allows for barycenter quadrature on
  /// complex polyhedrons. Note: barycenter quadrature is exact for
  /// polynom deg <= 1.

  class BarycenterQuadrature
  {
  public:

    /// Create barycenter quadrature rule for given polyhedron
    BarycenterQuadrature(const Nef_polyhedron_3& polyhedron);

    /// Return points
    const std::vector<Point>& points() const
    { return _points; }

    /// Return weights
    const std::vector<double>& weights() const
    { return _weights; }

    /// Return number of quadrature points/weights
    uint size() const
    { assert(_points.size() == _weights.size()); return _points.size(); }

  private:

    ///Computes barycenter and weight.
    void compute_quadrature(const Nef_polyhedron_3 &);

    std::vector<Point> _points;
    std::vector<double> _weights;

  };

}

#endif

#endif
