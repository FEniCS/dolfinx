// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-14
// Last changed: 2005-10-31

#include <dolfin.h>

using namespace dolfin;

/// Test problem taken from "Multirate time stepping for parabolic PDEs"
/// by Savcenco, Hundsdorfer and Verwer:
///
///    u' - epsilon u'' = gamma u^2 (1 - u)  in (0, L) x (0, T).
///
/// The solution is a reaction front sweeping across the domain.

class Reaction : public ODE
{
public:

  /// Constructor
  Reaction(unsigned int N, real T, real L, real epsilon, real gamma)
    : ODE(N, T), L(L), epsilon(epsilon), gamma(gamma)
  {
    // Compute parameters
    h = L / static_cast<real>(N - 1);
    lambda = 0.5*sqrt(2.0*gamma/epsilon);
    
    // Set sparse dependency pattern
    for (unsigned int i = 0; i < N; i++)
    {
      dependencies.setsize(i, 3);
      if ( i > 0 ) dependencies.set(i, i - 1);
      dependencies.set(i, i);
      if ( i < (N - 1) ) dependencies.set(i, i + 1);
    }
  }

  /// Initial condition
  real u0(unsigned int i)
  {
    const real x = static_cast<real>(i)*h;
    return 1.0 / (1.0 + exp(lambda*(x - 1.0)));
  }

  /// Right-hand side, mono-adaptive version
  void f(const real u[], real t, real y[])
  {
    for (unsigned int i = 0; i < N; i++)
    {
      const real ui = u[i];

      real sum = 0.0;
      if ( i == 0 )
	sum = u[i + 1] - ui;
      else if ( i == (N - 1) )
	sum = u[i - 1] - ui;
      else
	sum = u[i + 1] - 2.0*ui + u[i - 1];

      y[i] = epsilon * sum / (h*h) + gamma * ui*ui * (1.0 - ui);
    }
  }

  /// Right-hand side, multi-adaptive version
  real f(const real u[], real t, unsigned int i)
  {
    const real ui = u[i];
    
    real sum = 0.0;
    if ( i == 0 )
      sum = u[i + 1] - ui;
    else if ( i == (N - 1) )
      sum = u[i - 1] - ui;
    else
      sum = u[i + 1] - 2.0*ui + u[i - 1];
    
    return epsilon * sum / (h*h) + gamma * ui*ui * (1.0 - ui);
  }

public:

  real L;       // Length of domain
  real epsilon; // Diffusivity
  real gamma;   // Reaction rate
  real h;       // Mesh size
  real lambda;  // Parameter for initial data

};

int main(int argc, char* argv[])
{
  // Parse command line arguments
  if ( argc != 3 )
  {
    dolfin_info("Usage: dolfin-reaction method TOL");
    dolfin_info("");
    dolfin_info("method - 'cg' or 'mcg'");
    dolfin_info("TOL    - tolerance");
    return 1;
  }
  const char* method = argv[1];
  const real TOL = static_cast<real>(atof(argv[2]));

  dolfin_set("solver", "newton");
  dolfin_set("maximum time step", 0.1);
  dolfin_set("tolerance", TOL);
  dolfin_set("method", method);
  dolfin_set("order", 1);
  dolfin_set("save final solution", true);


  dolfin_set("order", 0);
  dolfin_set("save solution", true);

  //dolfin_set("initial time step", 2.5e-3);

  //dolfin_set("fixed time step", true);
  //dolfin_set("partitioning threshold", 0.25);
  //dolfin_set("monitor convergence", true);
  //dolfin_set("discrete tolerance", 1e-10);

  // Uncomment to compute reference solution
  /*
    dolfin_set("save solution", false);
    dolfin_set("save final solution", true);
    dolfin_set("fixed time step", true);
    dolfin_set("initial time step", 0.00005);
    dolfin_set("discrete tolerance", 1e-14);
    dolfin_set("method", "cg");
    dolfin_set("order", 3);
  */
  
  Reaction ode(1000, 3.0, 5.0, 0.01, 100.0);
  ode.solve();

  return 0;
}
