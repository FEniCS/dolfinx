// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __HOMOTOPY_H
#define __HOMOTOPY_H

#include <stdio.h>
#include <dolfin/constants.h>
#include <dolfin/NewGMRES.h>
#include <dolfin/NewVector.h>

namespace dolfin
{

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

    /// Compute y = F(z)
    virtual void F(const complex z[], complex y[]) = 0;

    /// Compute y = F'(z) x
    virtual void JF(const complex z[], const complex x[], complex y[]) = 0;

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
    void feval(NewVector& F, ComplexODE& ode);

    uint n;          // Size of system
    uint M;          // Number of paths
    uint maxiter;    // Maximum number of iterations
    real tol;        // Tolerance for Newton's method
    NewGMRES solver; // GMRES solver
    FILE* fp;        // File pointer for saving solution
    uint* mi;        // Array of local path numbers
    complex* ci;     // Array of constants for system G(z) = 0
    NewVector x;     // Real-valued vector x corresponding to solution z of F(z) = 0 

  };

}

#endif
