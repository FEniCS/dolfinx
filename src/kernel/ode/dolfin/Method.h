// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __METHOD_H
#define __METHOD_H

#include <dolfin/Lagrange.h>
#include <dolfin/constants.h>

namespace dolfin
{

  class Lagrange;
  
  /// Base class for cGqMethod and dGqMethod, which contain all numeric constants,
  /// such as nodal points and nodal weights, needed for the method.
  
  class Method
  {
  public:
    
    /// Constructor
    Method(unsigned int q);

    /// Destructor
    virtual ~Method();

    /// Return number of stages (inline optimized)
    inline unsigned int stages() const { return m; }
    
    /// Return number of points (inline optimized)
    inline unsigned int size() const { return n; }
    
    /// Return degree (inline optimized)
    inline unsigned int degree() const { return q; }

    /// Return nodal point (inline optimized)
    inline real point(unsigned int i) const { return points[i]; }
    
    /// Return nodal weight j for dof i, including quadrature (inline optimized)
    inline real weight(unsigned int i, unsigned int j) const { return weights[i][j]; }

    /// Return sum of nodal weights for dof i, including quadrature (inline optimized)
    inline real weightsum(unsigned int i) const { return weightsums[i]; }

    /// Return quadrature weight, including only quadrature (inline optimized)
    inline real weight(unsigned int i) const { return qweights[i]; }

    /// Evaluate of basis function i at a point t within [0,1] (inline optimized)
    inline real basis(unsigned int i, real t) const { return trial->eval(i, t); }

    /// Evaluate of derivative of basis function i at a point t within [0,1] (inline optimized)
    inline real derivative(unsigned int i, real t) const { return trial->ddx(i, t); }
    
    /// Evaluation of derivative of basis function i at t = 1 (inline optimized)
    inline real derivative(unsigned int i) const { return derivatives[i]; }

    /// Compute new time step based on the given residual
    virtual real timestep(real r, real tol, real kmax) const = 0;

    /// Display method data
    virtual void show() const = 0;

  protected:
    
    void init();

    virtual void computeQuadrature () = 0;
    virtual void computeBasis      () = 0;
    virtual void computeWeights    () = 0;

    void computeWeightSums();
    void computeDerivatives();

    unsigned int m; // Number of stages
    unsigned int q; // Polynomial order
    unsigned int n; // Number of nodal points

    real*  points;
    real** weights;
    real*  weightsums;
    real*  qweights;
    real*  derivatives;

    uint offset;

    Lagrange* trial;
    Lagrange* test;

  };

}

#endif
