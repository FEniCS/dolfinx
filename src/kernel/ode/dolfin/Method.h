// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __METHOD_H
#define __METHOD_H

#include <dolfin/constants.h>

namespace dolfin {

  class Lagrange;
  
  /// Base class for cGqMethod and dGqMethod, which contain all numeric constants,
  /// such as nodal points and nodal weights, needed for the method.
  
  class Method {
  public:
    
    /// Constructor
    Method(unsigned int q);

    /// Destructor
    virtual ~Method();
    
    /// Return number of points (inline optimized)
    inline unsigned int size() const { return n; }
    
    /// Return degree (inline optimized)
    inline unsigned int degree() const { return q; }

    /// Return nodal point (inline optimized)
    inline real point(unsigned int i) const { return points[i]; }
    
    /// Return nodal weight j for degree of freedom i, including quadrature (inline optimized)
    inline real weight(unsigned int i, unsigned int j) const { return weights[i][j]; }

    /// Return quadrature weight, including only quadrature (inline optimized)
    inline real weight(unsigned int i) const { return qweights[i]; }

    /// Evaluation of basis function i at given point t within [0,1] (inline optimized)
    inline real basis(unsigned int i, real t) const { return trial->eval(i, t); }

    /// Evaluation of derivative of basis function i at given point t within [0,1] (inline optimized)
    inline real derivative(unsigned int i, real t) const { return trial->ddx(i, t); }
    
    /// Evaluation of derivative of basis function i at t = 1 (inline optimized)
    inline real derivative(unsigned int i) const { return derivatives[i]; }
    
    /// Display method data
    virtual void show() const = 0;

  protected:
    
    void init();

    virtual void computeQuadrature () = 0;
    virtual void computeBasis      () = 0;
    virtual void computeWeights    () = 0;

    void computeDerivatives();

    unsigned int q; // Polynomial order
    unsigned int n; // Number of nodal points

    real*  points;
    real** weights;
    real*  qweights;
    real*  derivatives;

    Lagrange* trial;
    Lagrange* test;

  };

}

#endif
