// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_FUNCTION_H
#define __GENERIC_FUNCTION_H

#include <dolfin/ElementFunction.h>

namespace dolfin {

  class Cell;
  class Node;
  class Point;
  class Mesh;
  class ElementData;
  class FiniteElement;
  
  class GenericFunction {
  public:
    
    GenericFunction();
    virtual ~GenericFunction();

    // Evaluation of function
    virtual real operator() (const Node&  n, real t) const;
    virtual real operator() (const Node&  n, real t);
    virtual real operator() (const Point& p, real t) const;
    virtual real operator() (const Point& p, real t);
    virtual real operator() (real x, real y, real z, real t) const;
    virtual real operator() (real x, real y, real z, real t);
    virtual real operator() (unsigned int i, real t) const;
    virtual real operator() (unsigned int i, real t);

    // Update function to given time
    virtual void update(real t);

    // Return current time
    real time() const;

    // FIXME: Special member functions below: Should they be removed?
    //---------------------------------------------------------------

    // Return the mesh
    virtual Mesh& mesh() const;
    
    // Get element data
    virtual ElementData& elmdata();

    // Update values of element function
    virtual void update(FunctionSpace::ElementFunction &v,
			const FiniteElement &element, 
			const Cell &cell, real t) const;
    
  };
  
}

#endif
