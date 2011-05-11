// Copyright (C) 2005-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-05-02
// Last changed: 2009-09-08

#ifndef __METHOD_H
#define __METHOD_H

#include <dolfin/math/Lagrange.h>
#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class Lagrange;

  /// Base class for cGqMethod and dGqMethod, which contain all numeric constants,
  /// such as nodal points and nodal weights, needed for the method.

  class Method : public Variable
  {
  public:

    enum Type {cG, dG, none};

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

    /// Update solution values using fixed-point iteration
    void update(real x0, real f[], real k, real values[]) const;

    /// Update solution values using fixed-point iteration (damped version)
    void update(real x0, real f[], real k, real values[], real alpha) const;

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

    /// Get nodal values
    virtual void get_nodal_values(const real& x0, const real* x, real* nodal_values) const = 0;

    /// Get trial functions
    inline const Lagrange get_trial() const { return *trial; }

    /// Get quadrature weights
    inline const real* get_quadrature_weights() const {return qweights; }

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

  protected:

    void init();

    virtual void compute_quadrature () = 0;
    virtual void compute_basis      () = 0;
    virtual void compute_weights    () = 0;

    void compute_derivatives();

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
