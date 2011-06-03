// Copyright (C) 2003-2006 Anders Logg
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
// First added:  2003-02-06
// Last changed: 2009-08-10

#ifndef __QUADRATURE_H
#define __QUADRATURE_H

#include <vector>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class Quadrature : public Variable
  {
  public:

    /// Constructor
    Quadrature(unsigned int n, double m=1.0);

    /// Destructor
    virtual ~Quadrature();

    /// Return number of quadrature points
    int size() const;

    /// Return quadrature point
    double point(unsigned int i) const;

    /// Return quadrature weight
    double weight(unsigned int i) const;

    /// Return sum of weights (length, area, volume)
    double measure() const;

  protected:

    // Quadrature points
    std::vector<double> points;

    // Quadrature weights
    std::vector<double> weights;

  private:

    // Sum of weights
    double m;

  };

}

#endif
