// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2005-05-17
// Last changed: 2006-12-06

#ifndef __FEBASIS_H
#define __FEBASIS_H

#include <dolfin/constants.h>
#include <dolfin/Point.h>
#include <dolfin/Cell.h>
#include <dolfin/DenseMatrix.h>
#include <dolfin/Array.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Function.h>
#include <dolfin/NewAffineMap.h>

namespace dolfin
{
  class FEBasis
  {
  public:
    
    /// Constructor
    FEBasis();

    /// Destructor
    ~FEBasis();

    void construct(FiniteElement& element);

    real evalPhysical(Function& f, Point& p, NewAffineMap& map,
		      dolfin::uint i);

    Array<Function *> functions;

    FiniteElementSpec spec;
  };

}

#endif
