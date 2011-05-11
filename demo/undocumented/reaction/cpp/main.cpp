// Copyright (C) 2005-2009 Anders Logg
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
// First added:  2005-10-14
// Last changed: 2009-09-08

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
      if (i > 0) dependencies.set(i, i - 1);
      dependencies.set(i, i);
      if (i < (N - 1)) dependencies.set(i, i + 1);
    }
  }

  /// Initial condition
  void u0(Array<real>& u)
  {
    for (unsigned int i = 0; i < N; i++)
    {
      const real x = static_cast<real>(i)*h;
      u[i] = 1.0 / (1.0 + real_exp(lambda*(x - 1.0)));
    }
  }

  /// Right-hand side, multi-adaptive version
  real f(const Array<real>& u, real t, unsigned int i)
  {
    const real ui = u[i];

    real sum = 0.0;
    if ( i == 0 )
      sum = u[i + 1] - ui;
    else if (i == (N - 1))
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
  real v;       // Speed of reaction front

};

int main()
{
  info("Reaction ODE demo needs to be fixed.");

  // FIXME: Does not work with GMP enabled

  // Set some parameters
  const real T = 0.01;
  const real epsilon = 0.01;
  const real gamma = 1000.0;
  const real L = 1.0;
  const unsigned int N = 5000;

  // Create ODE
  Reaction ode(N, T, L, epsilon, gamma);
  ode.parameters["method"] = "mcg";
  ode.parameters["order"] = 1;
  ode.parameters["nonlinear_solver"] = "fixed-point";
  ode.parameters["tolerance"] = 1e-3;
  ode.parameters["partitioning_threshold"] = 0.5;
  ode.parameters["initial_time_step"] = 1e-5;
  ode.parameters["maximum_time_step"] = 1e-3;
  ode.parameters["adaptive_samples"] = true;

  // Solve ODE
  ode.solve();

  return 0;
}
