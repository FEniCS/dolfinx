// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __METHOD_H
#define __METHOD_H

#include <dolfin/constants.h>

namespace dolfin {

  /// Base class for cGqMethod and dGqMethod, which contain all numeric constants,
  /// such as nodal points and nodal weights, needed for the method.
  
  class Method {
  public:
    
    /// Constructor
    Method(int q);

    /// Destructor
    ~Method();
    
    /// Return number of points
    int size() const;
    
    /// Return degree
    int degree() const;

    /// Return nodal point
    real point(int i) const;
    
    /// Return nodal weight (including quadrature and weight function)
    real weight(int i) const;

    /// Return quadrature weight (including only quadrature)
    real qweight(int i) const;

  protected:
    
    void init();

    virtual void computeQuadrature () = 0;
    virtual void computeBasis      () = 0;
    virtual void computeWeights    () = 0;

    int q;
    int n;

    real* points;
    real* weights;
    real* qweights;

    Lagrange* trial;
    Lagrange* test;

  };

}

#endif
