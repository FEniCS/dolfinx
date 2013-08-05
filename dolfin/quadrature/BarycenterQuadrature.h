// Copyright (C) 2010 Andre Massing
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
// Modified by Anders Logg, 2010.
//
// First added:  2010-03-17
// Last changed: 2010-04-12

#ifndef  __BARYCENTER_QUADRATURE_H
#define  __BARYCENTER_QUADRATURE_H

#ifdef HAS_CGAL

#include <vector>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <dolfin/geometry/Point.h>

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

    /// Create barycenter quadrature rule for given CGAL polyhedron
    BarycenterQuadrature(const Nef_polyhedron_3& polyhedron);

    /// Return points
    const std::vector<Point>& points() const
    { return _points; }

    /// Return weights
    const std::vector<double>& weights() const
    { return _weights; }

    /// Return number of quadrature points/weights
    std::size_t size() const
    { dolfin_assert(_points.size() == _weights.size()); return _points.size(); }

  private:

    ///Computes barycenter and weight.
    void compute_quadrature(const Nef_polyhedron_3 &);

    std::vector<Point> _points;
    std::vector<double> _weights;

  };

}

#endif

#endif
