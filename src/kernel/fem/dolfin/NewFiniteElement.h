// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_FINITE_ELEMENT_H
#define __NEW_FINITE_ELEMENT_H

#include <dolfin/Cell.h>
#include <dolfin/Node.h>
#include <dolfin/Point.h>

namespace dolfin
{
  
  /// This is a temporary implementation of the representation of a
  /// finite element, providing only the information necessary to do
  /// assembly.

  class NewFiniteElement
  {
  public:
   
    /// Constructor
    NewFiniteElement();

    /// Destructor
    virtual ~NewFiniteElement();
    
    /// Return dimension of the finite element space
    virtual unsigned int spacedim() const = 0;

    /// Return dimension of the underlying shape
    virtual unsigned int shapedim() const = 0;

    /// Return vector dimension of the finite element space
    virtual unsigned int tensordim(unsigned int i) const = 0;

    /// Return vector dimension of the finite element space
    virtual unsigned int rank() const = 0;

    /// Return map from local to global degree of freedom
    virtual unsigned int dof(unsigned int i, const Cell& cell) const = 0;
    
    /// Return map from local degree of freedom to global coordinate
    virtual const Point& coord(unsigned int i, const Cell& cell) const = 0;

  };

}

#endif
