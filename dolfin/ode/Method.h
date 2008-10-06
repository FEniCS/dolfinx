// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-05-02
// Last changed: 2006-07-07

#ifndef __METHOD_H
#define __METHOD_H

#include <dolfin/math/Lagrange.h>
#include <dolfin/common/types.h>

namespace dolfin
{

  class Lagrange;
  class uBLASVector;
  
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
    inline double npoint(unsigned int i) const { return npoints[i]; }
    
    /// Return quadrature point (inline optimized)
    inline double qpoint(unsigned int i) const { return qpoints[i]; }
    
    /// Return nodal weight j for node i, including quadrature (inline optimized)
    inline double nweight(unsigned int i, unsigned int j) const { return nweights[i][j]; }

    /// Return quadrature weight, including only quadrature (inline optimized)
    inline double qweight(unsigned int i) const { return qweights[i]; }

    /// Evaluation of trial space basis function i at given tau (inline optimized)
    inline double eval(unsigned int i, double tau) const { return trial->eval(i, tau); }
    
    /// Evaluation of derivative of basis function i at t = 1 (inline optimized)
    inline double derivative(unsigned int i) const { return derivatives[i]; }

    /// Update solution values using fixed-point iteration
    void update(double x0, double f[], double k, double values[]) const;

    /// Update solution values using fixed-point iteration (damped version)
    void update(double x0, double f[], double k, double values[], double alpha) const;

    /// Evaluate solution at given point
    virtual double ueval(double x0, double values[], double tau) const = 0;

    /// Evaluate solution at given point
    virtual double ueval(double x0, uBLASVector& values, uint offset, double tau) const = 0;

    /// Evaluate solution at given node
    virtual double ueval(double x0, double values[], uint i) const = 0;

    /// Compute residual at right end-point
    virtual double residual(double x0, double values[], double f, double k) const = 0;

    /// Compute residual at right end-point
    virtual double residual(double x0, uBLASVector& values, uint offset, double f, double k) const = 0;
  
    /// Compute new time step based on the given residual
    virtual double timestep(double r, double tol, double k0, double kmax) const = 0;

    /// Compute error estimate (modulo stability factor)
    virtual double error(double k, double r) const = 0;
    
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

    double*  qpoints;     // Quadrature points
    double*  npoints;     // Nodal points
    double*  qweights;    // Quadrature weights
    double** nweights;    // Nodal weights
    double*  derivatives; // Weights for derivative at end-point

    Lagrange* trial;
    Lagrange* test;

    Type _type;

  };

}

#endif
