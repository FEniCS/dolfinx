// Copyright (C) 2004 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.

#ifndef __ADAPTIVE_ITERATION_H
#define __ADAPTIVE_ITERATION_H

#include <dolfin/NewArray.h>
#include <dolfin/Iteration.h>

namespace dolfin
{

  /// State-specific behavior of fixed point iteration for general stiff problems.

  class AdaptiveIteration : public Iteration
  {
  public:

    AdaptiveIteration(Solution& u, RHS& f, FixedPointIteration& fixpoint,
		      unsigned int maxiter, real maxdiv, real maxconv, real tol);
    
    ~AdaptiveIteration();
    
    State state() const;

    void start(TimeSlab& timeslab);
    void start(NewArray<Element*>& elements);
    void start(Element& element);

    void update(TimeSlab& timeslab, const Damping& d);
    void update(NewArray<Element*>& elements, const Damping& d);
    void update(Element& element, const Damping& d);
    
    void stabilize(TimeSlab& timeslab, const Residuals& r, Damping& d);
    void stabilize(NewArray<Element*>& elements, const Residuals& r, Damping& d);
    void stabilize(Element& element, const Residuals& r, Damping& d);
    
    bool converged(TimeSlab& timeslab, Residuals& r, unsigned int n);
    bool converged(NewArray<Element*>& elements, Residuals& r, unsigned int n);
    bool converged(Element& element, Residuals& r, unsigned int n);

    bool diverged(TimeSlab& timeslab, Residuals& r, unsigned int n, Iteration::State& newstate);
    bool diverged(NewArray<Element*>& elements, Residuals& r, unsigned int n, Iteration::State& newstate);
    bool diverged(Element& element, Residuals& r, unsigned int n, Iteration::State& newstate);

    void report() const;

  private:

    // Type of iteration
    enum Method {gauss_jacobi, gauss_seidel};

    // Additional data for Gauss-Jacobi iteration
    struct Values
    {
      Values();
      ~Values();

      void init(unsigned int size);

      real* values;
      unsigned int size;
      unsigned int offset;
    };

    // Gauss-Jacobi iteration on element list
    void updateGaussJacobi(NewArray<Element*>& elements, const Damping& d);

    // Gauss-Seidel iteration on element list
    void updateGaussSeidel(NewArray<Element*>& elements, const Damping& d);
    
    // Compute divergence
    real computeDivergence(const Residuals& r) const;

    // Compute alpha
    real computeAlpha(real rho) const;

    // Compute number of damping steps
    unsigned int computeSteps(real rho) const;

    // Initialize additional data
    void initData(const NewArray<Element*>& elements);

    // Copy data to element list
    void copyData(NewArray<Element*>& elements) const;

    // Compute size of data
    unsigned int dataSize(const NewArray<Element*>& elements) const;

    // Current method (Gauss-Jacobi or Gauss-Seidel)
    Method method;
    
    // Additional data for Gauss-Jacobi iteration
    Values values;

    // Number of stabilizing iterations
    unsigned int m;

    // Number of remaining stabilizing iterations
    unsigned int j;
    
    // Stabilization parameter
    real alpha;

    // Angle of sector, gamma = cos(theta)
    real gamma;

  };

}

#endif
