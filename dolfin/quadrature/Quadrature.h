// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-02-06
// Last changed: 2009-08-10

#ifndef __QUADRATURE_H
#define __QUADRATURE_H

#include <vector>
#include <dolfin/common/real.h>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class Quadrature : public Variable
  {
  public:

    /// Constructor
    Quadrature(unsigned int n, real m=1.0);

    /// Destructor
    virtual ~Quadrature();

    /// Return number of quadrature points
    int size() const;

    /// Return quadrature point
    real point(unsigned int i) const;

    /// Return quadrature weight
    real weight(unsigned int i) const;

    /// Return sum of weights (length, area, volume)
    real measure() const;

  protected:

    // Quadrature points
    std::vector<real> points;

    // Quadrature weights
    std::vector<real> weights;

  private:

    // Sum of weights
    real m;

  };

}

#endif
