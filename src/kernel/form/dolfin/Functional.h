// Copyright (C) 2005 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-02

#ifndef __FUNCTIONAL_H
#define __FUNCTIONAL_H

#include <dolfin/Form.h>

namespace dolfin
{

  /// Functional represents a functional F(u) 

  class Functional : public Form
  {
  public:
    
    /// Constructor
    Functional(uint num_functions = 0);
    
    /// Destructor
    virtual ~Functional();

    /// Compute element vector (interior contribution)
    virtual void eval(real block[], const AffineMap& map) const;

    /// Compute element vector (boundary contribution)
    virtual void eval(real block[], const AffineMap& map, uint segment) const;

    /// Return finite element defining the test space
    const FiniteElement& test() const;

  protected:

    // Finite element defining the test space
    FiniteElement* _test;

  };

}

#endif
