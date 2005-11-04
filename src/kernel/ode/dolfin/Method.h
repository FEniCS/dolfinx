// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-05-02
// Last changed: 2005-11-02

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
    
    enum Type { cG, dG, none };

    /// Constructor
    Method(unsigned int q, unsigned int nq, unsigned int nn);

    /// Destructor
    virtual ~Method();

    /// Return type (inline optimized)
    inline Type type() const { return _type; }

    /// Return degree (inline optimized)
    inline unsigned int degree() const { return q; }

    /// Return order (inline optimized)
    virtual unsigned int order() const { return p; }

    /// Return number of nodal points (inline optimized)
    inline unsigned int nsize() const { return nn; }

    /// Return number of quadrature points (inline optimized)
    inline unsigned int qsize() const { return nq; }

    /// Return nodal point (inline optimized)
    inline real npoint(unsigned int i) const { return npoints[i]; }
    
    /// Return quadrature point (inline optimized)
    inline real qpoint(unsigned int i) const { return qpoints[i]; }
    
    /// Return nodal weight j for node i, including quadrature (inline optimized)
    inline real nweight(unsigned int i, unsigned int j) const { return nweights[i][j]; }

    /// Return quadrature weight, including only quadrature (inline optimized)
    inline real qweight(unsigned int i) const { return qweights[i]; }

    /// Evaluation of trial space basis function i at given tau (inline optimized)
    inline real eval(unsigned int i, real tau) const { return trial->eval(i, tau); }
    
    /// Evaluation of derivative of basis function i at t = 1 (inline optimized)
    inline real derivative(unsigned int i) const { return derivatives[i]; }

    /// Update solution values using fixed point iteration
    void update(real x0, real f[], real k, real values[]) const;

    /// Evaluate solution at given point
    virtual real ueval(real x0, real values[], real tau) const = 0;

    /// Evaluate solution at given node
    virtual real ueval(real x0, real values[], uint i) const = 0;

    /// Compute residual at right end-point
    virtual real residual(real x0, real values[], real f, real k) const = 0;
  
    /// Compute new time step based on the given residual
    virtual real timestep(real r, real tol, real k0, real kmax) const = 0;

    /// Compute error estimate (modulo stability factor)
    virtual real error(real k, real r) const = 0;
    
    /// Display method data
    virtual void disp() const = 0;

  protected:
    
    void init();

    virtual void computeQuadrature () = 0;
    virtual void computeBasis      () = 0;
    virtual void computeWeights    () = 0;

    void computeDerivatives();

    unsigned int q;  // Polynomial degree
    unsigned int p;  // Convergence order
    unsigned int nq; // Number of quadrature points
    unsigned int nn; // Number of nodal points

    real*  qpoints;     // Quadrature points
    real*  npoints;     // Nodal points
    real*  qweights;    // Quadrature weights
    real** nweights;    // Nodal weights
    real*  derivatives; // Weights for derivative at end-point

    Lagrange* trial;
    Lagrange* test;

    Type _type;

  };

}

#endif
