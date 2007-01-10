// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-09-18
// Last changed: 2006-12-12

#ifndef __FUNCTIONAL_H
#define __FUNCTIONAL_H

#include <dolfin/Form.h>

namespace dolfin
{

  /// Functional represents a multilinear form of the type
  ///
  ///     M = M(w1, w2, ..., wn)
  ///
  /// where w1, w2, ..., wn are any given functions.

  class Functional : public Form
  {
  public:
    
    /// Constructor
    Functional(uint num_functions = 0);
    
    /// Destructor
    virtual ~Functional();

    /// Check if there is a contribution from the interior
    virtual bool interior_contribution() const = 0;

    /// Compute element vector (interior contribution)
    virtual void eval(real block[], const AffineMap& map, real det) const = 0;

    /// Check if there is a contribution from the boundary
    virtual bool boundary_contribution() const = 0;

    /// Compute element vector (boundary contribution)
    virtual void eval(real block[], const AffineMap& map, real det, uint segment) const = 0;

    /// Check if there is a contribution from the interior boundary
    virtual bool interior_boundary_contribution() const = 0;

    /// Compute exterior facet tensor
    virtual void eval(real block[],
                      const AffineMap& map0, const AffineMap& map1, real det,
                      uint facet0, uint facet1, uint alignment) const = 0;

  protected:
    
    // Update local data structures
    void updateLocalData();

  };

}

#endif
