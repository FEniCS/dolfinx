// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-14
// Last changed: 2005-12-20

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
    v = 0.5*sqrt(2.0*gamma*epsilon);
    
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

  /// Specify time step, mono-adaptive version (used for testing)
  real timestep(real t, real k0) const
  {
    return 1e-5;
  }

  /// Specify time step, mono-adaptive version (used for testing)
  real timestep(real t, unsigned int i, real k0) const
  {
    const real w  = 0.1;
    const real v  = 2.22;
    const real x  = static_cast<real>(i)*h;
    if ( fabs(x - 1.0 - v*t) < w )
      return 1e-5;
    else
      return 1e-3;
  }

public:

  real L;       // Length of domain
  real epsilon; // Diffusivity
  real gamma;   // Reaction rate
  real h;       // Mesh size
  real lambda;  // Parameter for initial data
  real v;       // Speed of reaction front

};

int main(int argc, char* argv[])
{
  // Parse command line arguments
  if ( argc != 11 )
  {
    dolfin_info("Usage: dolfin-reaction method solver TOL k0 kmax T gamma");
    dolfin_info("");
    dolfin_info("method - 'cg' or 'mcg'");
    dolfin_info("solver - 'fixed-point' or 'newton'");
    dolfin_info("TOL    - tolerance");
    dolfin_info("k0     - initial time step");
    dolfin_info("kmax   - initial time step");
    dolfin_info("T      - final time");
    dolfin_info("gamma  - reaction rate, something like 100.0 or 1000.0");
    dolfin_info("N      - number of components");
    dolfin_info("L      - length of domain");
    dolfin_info("params - a DOLFIN parameter file for other parameters");
    return 1;
  }
  const char* method = argv[1];
  const char* solver = argv[2];
  const real TOL = static_cast<real>(atof(argv[3]));
  const real k0 = static_cast<real>(atof(argv[4]));
  const real kmax = static_cast<real>(atof(argv[5]));
  const real T = static_cast<real>(atof(argv[6]));
  const real gamma = static_cast<real>(atof(argv[7]));
  const unsigned int N = static_cast<unsigned int>(atoi(argv[8]));
  const real L = static_cast<unsigned int>(atof(argv[9]));
  const char* params = argv[10];
  
  // Load parameters from file
  File file(params);
  file >> ParameterSystem::parameters;

  // Set solver parameters
  set("method", method);
  set("solver", solver);
  set("order", 1);
  set("tolerance", TOL);
  set("initial time step", k0);
  set("maximum time step", kmax);
//   set("save solution", false);
  set("save final solution", true);
//   set("partitioning threshold", 0.7);
//   set("partitioning threshold", 1e-5);
  
  // Need to save in Python format for plot_reaction.py to work
  //set("ode solution file name", "primal.py");
  //set("save solution", true);

  //set("adaptive samples", true);
  //set("monitor convergence", true);
//   set("fixed time step", true);
  //set("discrete tolerance", 1e-10);
  //set("diagonal newton damping", true);
  //set("updated jacobian", true);
  
  // Uncomment to compute reference solution
  /*
    set("save solution", false);
    set("save final solution", true);
    set("fixed time step", true);
    set("initial time step", 0.0001);
    set("discrete tolerance", 1e-14);
    set("method", "cg");
    set("order", 3);
  */
  
  //Reaction ode(10, 3.0, 5.0, 0.01, 100.0);
  //Reaction ode(100, 3.0, 5.0, 0.01, 100.0);
  //Reaction ode(1000, 0.5, 5.0, 0.01, 100.0);

  Reaction ode(N, T, L, 0.01, gamma);
  ode.solve();

  return 0;
}
