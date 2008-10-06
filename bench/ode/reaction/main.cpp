// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-10-14
// Last changed: 2006-08-22

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
      if ( i > 0 ) dependencies.set(i, i - 1);
      dependencies.set(i, i);
      if ( i < (N - 1) ) dependencies.set(i, i + 1);
    }
  }

  /// Initial condition
  void u0(uBLASVector& u)
  {
    for (unsigned int i = 0; i < N; i++)
    {
      const double x = static_cast<double>(i)*h;
      u(i) = 1.0 / (1.0 + exp(lambda*(x - 1.0)));
    }
  }

  /// Right-hand side, mono-adaptive version
  void f(const uBLASVector& u, double t, uBLASVector& y)
  {
    printf("MONO!\n");
    for (unsigned int i = 0; i < N; i++)
    {
      const double ui = u(i);

      double sum = 0.0;
      if ( i == 0 )
	sum = u(i + 1) - ui;
      else if ( i == (N - 1) )
	sum = u(i - 1) - ui;
      else
	sum = u(i + 1) - 2.0*ui + u(i - 1);

      y(i) = epsilon * sum / (h*h) + gamma * ui*ui * (1.0 - ui);
    }
  }

  /// Right-hand side, multi-adaptive version
  double f(const uBLASVector& u, double t, unsigned int i)
  {
    printf("MULTI!\n");
    const double ui = u(i);
    
    double sum = 0.0;
    if ( i == 0 )
      sum = u(i + 1) - ui;
    else if ( i == (N - 1) )
      sum = u(i - 1) - ui;
    else
      sum = u(i + 1) - 2.0*ui + u(i - 1);
    
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

int main(int argc, char* argv[])
{
  // Parse command line arguments
  if ( argc != 7 )
  {
    message("Usage: dolfin-ode-reaction method solver tol kmax N L params");
    message("");
    message("method - 'cg' or 'mcg'");
    message("solver - 'fixed-point' or 'newton'");
    message("tol    - tolerance");
    message("N      - number of components");
    message("L      - length of domain");
    message("params - name of parameter file");
    return 1;
  }
  const char* method = argv[1];
  const char* solver = argv[2];
  const double tol = static_cast<double>(atof(argv[3]));
  const unsigned int N = static_cast<unsigned int>(atoi(argv[4]));
  const double L = static_cast<unsigned int>(atof(argv[5]));
  const char* params = argv[6];
  
  // Load solver parameters from file
  File file(params);
  file >> ParameterSystem::parameters;

  // Set remaining solver parameters from command-line arguments
  set("ODE method", method);
  set("ODE nonlinear solver", solver);
  set("ODE tolerance", tol);

  // Set fixed parameters for test problem
  const double T = 1.0;
  const double epsilon = 0.01;
  const double gamma = 1000.0;

  // Solve system of ODEs
  Reaction ode(N, T, L, epsilon, gamma);
  ode.solve();

  return 0;
}
