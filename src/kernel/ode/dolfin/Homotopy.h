// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __HOMOTOPY_H
#define __HOMOTOPY_H

#include <stdio.h>
#include <dolfin/constants.h>
#include <dolfin/LU.h>
#include <dolfin/Vector.h>

namespace dolfin
{

  class LinearSolver;
  class HomotopyODE;
  class ComplexODE;

  class Homotopy
  {
  public:
    
    /// Create homotopy for system of given size
    Homotopy(uint n);

    /// Destructor
    virtual ~Homotopy();

    /// Solve homotopy
    void solve();

    /// Return initial value (solution of G(z) = 0), optional
    virtual complex z0(uint i);

    /// Compute y = F(z)
    virtual void F(const complex z[], complex y[]) = 0;

    /// Compute y = F'(z) x
    virtual void JF(const complex z[], const complex x[], complex y[]) = 0;

    /// Compute y = G(z), optional
    virtual void G(const complex z[], complex y[]);

    /// Compute y = G'(z) x, optional
    virtual void JG(const complex z[], const complex x[], complex y[]);

    /// Return degree of polynomial F_i(z)
    virtual uint degree(uint i) const = 0;

    /// Friends
    friend class HomotopyODE;

  private:

    // Count the number of paths
    uint countPaths() const;

    // Compute component path numbers from global path number
    void computePath(uint m);

    // Compute solution with Newton's method from current starting point
    void computeSolution(HomotopyODE& ode);

    // Save solution to file
    void saveSolution();

    // Randomize system G(z) = 0
    void randomize();

    // Evaluate right-hand side
    void feval(Vector& F, ComplexODE& ode);

    uint n;                  // Size of system
    uint M;                  // Number of paths
    uint maxiter;            // Maximum number of iterations
    real tol;                // Tolerance for Newton's method
    real divtol;             // Tolerance for divergence of homotopy path
    bool monitor;            // True if we should monitor the homotopy
    bool random;             // True if we should choose random initial data
    LinearSolver* solver; // GMRES solver
    FILE* fp;                // File pointer for saving solution
    uint* mi;                // Array of local path numbers
    complex* ci;             // Array of constants for system G(z) = 0
    Vector x;             // Real-valued vector x corresponding to solution z of F(z) = 0

  };

}

#endif
