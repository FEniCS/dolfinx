// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-14
// Last changed: 2005-10-14

#include <dolfin.h>

using namespace dolfin;

/// Test problem taken from "..." by ...:
///
///    u' - u'' = ...      in (0,1) x (0,T)
///
/// with T = ... The solution is a reaction front sweeping across
/// the domain.

class Reaction : public ODE
{
public:

  /// Constructor
  Reaction(unsigned int n) : ODE(n, 1.0), h(1.0/n)
  {
    // Set sparse dependency pattern
    for (unsigned int i = 0; i < n; i++)
    {
      dependencies.setsize(i, 3);
      if ( i > 0 ) dependencies.set(i, i - 1);
      dependencies.set(i, i);
      if ( i < n ) dependencies.set(i, i + 1);
    }
  }

  /// Initial condition
  real u0(unsigned int i)
  {
    const real x = static_cast<real>(i)*h;
    return x;
  }

  /// Right-hand side, mono-adaptive version
  void f(const real u[], real t, real y[])
  {
    for (unsigned int i = 0; i < N; i++)
    {
      real sum = -2.0*u[i];
      if ( i > 0 ) sum += u[i - 1];
      if ( i < N ) sum += u[i + 1];

      y[i] = sum;
    }
  }

  /// Right-hand side, multi-adaptive version
  real f(const real u[], real t, unsigned int i)
  {
    real sum = -2.0*u[i];
    if ( i > 0 ) sum += u[i - 1];
    if ( i < N ) sum += u[i + 1];
    
    return sum;
  }
  
private:

  real h; // Mesh size

};

int main()
{
  //dolfin_set("method", "cg");
  //dolfin_set("fixed time step", true);
  //dolfin_set("discrete tolerance", 0.01);
  //dolfin_set("partitioning threshold", 0.25);
  //dolfin_set("maximum local iterations", 2);
  
  // Uncomment this to run benchmarks
  //dolfin_set("save solution", false);
  
  Reaction ode(100);
  ode.solve();

  return 0;
}
