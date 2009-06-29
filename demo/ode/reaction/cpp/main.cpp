// Copyright (C) 2005-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-10-14
// Last changed: 2009-06-29

#include <dolfin.h>

using namespace dolfin;

/// Test problem taken from "Multirate time stepping for parabolic PDEs"
/// by Savcenco, Hundsdorfer and Verwer:
///
///    u' - epsilon u'' = gamma u^2 (1 - u)  in (0, L) x (0, T).
///
/// The solution is a reaction front sweeping across the domain.
///
/// This is a simplified (only multi-adaptive) version of the ODE
/// benchmark found under bench/ode/reaction

class Reaction : public ODE
{
public:

  /// Constructor
  Reaction(unsigned int N, double T, double L, double epsilon, double gamma)
    : ODE(N, T), L(L), epsilon(epsilon), gamma(gamma)
  {
    // Compute parameters
    h = L / static_cast<double>(N - 1);
    lambda = 0.5*sqrt(2.0*gamma/epsilon);
    v = 0.5*sqrt(2.0*gamma*epsilon);
    
    // Set sparse dependency pattern
    for (unsigned int i = 0; i < N; i++)
    {
      dependencies.setsize(i, 3);
      if (i > 0) dependencies.set(i, i - 1);
      dependencies.set(i, i);
      if (i < (N - 1)) dependencies.set(i, i + 1);
    }
  }

  /// Initial condition
  void u0(real* u)
  {
    for (unsigned int i = 0; i < N; i++)
    {
      const double x = static_cast<double>(i)*h;
      u[i] = 1.0 / (1.0 + exp(lambda*(x - 1.0)));
    }
  }

  /// Right-hand side, multi-adaptive version
  double f(const real* u, double t, unsigned int i)
  {
    const double ui = u[i];
    
    double sum = 0.0;
    if ( i == 0 )
      sum = u[i + 1] - ui;
    else if (i == (N - 1))
      sum = u[i - 1] - ui;
    else
      sum = u[i + 1] - 2.0*ui + u[i - 1];
    
    return epsilon * sum / (h*h) + gamma * ui*ui * (1.0 - ui);
  }

public:

  double L;       // Length of domain
  double epsilon; // Diffusivity
  double gamma;   // Reaction rate
  double h;       // Mesh size
  double lambda;  // Parameter for initial data
  double v;       // Speed of reaction front

};

int main()
{
  // Set some parameters
  const double T = 0.01;
  const double epsilon = 0.01;
  const double gamma = 1000.0;
  const double L = 1.0;
  const unsigned int N = 5000;

  // Set up ODE
  Reaction ode(N, T, L, epsilon, gamma);
  ode.set("ODE method", "mcg");
  ode.set("ODE order", 1);
  ode.set("ODE nonlinear solver", "fixed-point");
  ode.set("ODE tolerance", 1e-3);
  ode.set("ODE partitioning threshold", 0.5);
  ode.set("ODE initial time step", 1e-5);
  ode.set("ODE maximum time step", 1e-3);
  ode.set("ODE adaptive samples", true);

  // Solve ODE
  ode.solve();

  return 0;
}
