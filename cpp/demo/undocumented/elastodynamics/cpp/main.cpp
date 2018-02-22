// Copyright (C) 2009 Mirko Maraldi and Garth N. Wells
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2011
//
// First added:  2009-01-22
// Last changed: 2011-06-28

#include <dolfin.h>
#include "ElastoDynamics.h"
#include "DG0_eps_xx.h"

using namespace dolfin;

// External load
class Pressure : public Expression
{
public:

  Pressure(const double& t, const double& dt, bool old)
    : Expression(2), t(t), dt(dt), old(old) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    double time = t;
    if(old && time > 0.0)
      time -= dt;

    const double cutoff_time = 10.0*1.0/32.0;
    if (time < cutoff_time)
    {
      values[0] = 1.0*time/cutoff_time;
      values[1] = 0.0;
    }
    else
    {
      values[0] = 1.0;
      values[1] = 0.0;
    }
  }

private:

  const double& t;
  const double& dt;
  const bool old;
};

// Right boundary
class RightBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    if (1.0 - x[0] < DOLFIN_EPS && on_boundary)
      return true;
    else
      return false;
  }
};

class LeftBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    if (x[0] < DOLFIN_EPS)
      return true;
    else
      return false;
  }
};

// Acceleration update
void update_a(Function& a, const Function& u, const Function& a0,
              const Function& v0,  const Function& u0,
              double beta, double dt)
{
  // a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
  *a.vector()  = *u.vector();
  *a.vector() -= *u0.vector();
  *a.vector() *= 1.0/dt;
  *a.vector() -= *v0.vector();
  *a.vector() *= 1.0/((0.5-beta)*dt);
  *a.vector() -= *a0.vector();
  *a.vector() *= (0.5-beta)/beta;
}

// Velocity update
void update_v(Function& v, const Function& a, const Function& a0,
              const Function& v0, double gamma, double dt)
{
  // v = dt * ((1-gamma)*a0 + gamma*a) + v0
  *v.vector()  = *a0.vector();
  *v.vector() *= (1.0-gamma)/gamma;
  *v.vector() += *a.vector();
  *v.vector() *= dt*gamma;
  *v.vector() += *v0.vector();
}

int main(int argc, char* argv[])
{
  init(argc, argv);

  // Create mesh
  auto mesh = std::make_shared<Mesh>("../dolfin_fine.xml.gz");

  // Create function space
  auto V = std::make_shared<ElastoDynamics::FunctionSpace>(mesh);

  // Material parameters
  auto rho = std::make_shared<Constant>(1.0);  // mass density
  auto eta = std::make_shared<Constant>(0.25); // damping coefficient
  double E  = 1.0;                             // Youngs modulus
  double nu = 0.0;                             // Poisson ratio
  auto lambda = std::make_shared<Constant>((nu*E)/((1.0 + nu)*(1.0 - 2.0 * nu))); // Lame coefficient
  auto mu = std::make_shared<Constant>(E/(2.0*(1.0 + nu))); // Lame coefficient

  // Time stepping parameters
  auto alpha_m = std::make_shared<Constant>(0.2);
  auto alpha_f = std::make_shared<Constant>(0.4);
  auto beta = std::make_shared<Constant>(0.36);
  auto gamma = std::make_shared<Constant>(0.7);
  auto dt = std::make_shared<Constant>(1.0/32.0);    // time step
  double t = 0.0;           // initial time
  double T = 200*(*dt);        // final time

  // Body force
  auto f = std::make_shared<Constant>(0.0, 0.0);

  // External load
  RightBoundary right_boundary;
  auto right_boundary_function
    = std::make_shared<MeshFunction<std::size_t>>(mesh, 1);
  right_boundary.mark(*right_boundary_function, 3);
  auto p = std::make_shared<Pressure>(t, *dt, false);
  auto p0 = std::make_shared<Pressure>(t, *dt, true);

  // Dirichlet boundary conditions
  auto left_boundary = std::make_shared<LeftBoundary>();
  auto zero = std::make_shared<Constant>(0.0, 0.0);
  DirichletBC bc(V, zero, left_boundary);

  // Define solution vectors
  auto u = std::make_shared<Function>(V); // displacement
  auto u0 = std::make_shared<Function>(V);
  Function v(V);  // velocity
  auto v0 = std::make_shared<Function>(V);
  Function a(V);  // acceleration
  auto a0 = std::make_shared<Function>(V);

  // Set initial conditions and initialise acceleration function
  u0->vector()->zero();
  v0->vector()->zero();
  a0->vector()->zero();

  // Create forms
  ElastoDynamics::BilinearForm a_form(V, V);
  ElastoDynamics::LinearForm L(V);

  // Attach functions
  a_form.lmbda = lambda; a_form.mu = mu; a_form.rho = rho;
  a_form.eta = eta;
  a_form.beta = beta;
  a_form.gamma = gamma;
  a_form.dt = dt;
  a_form.alpha_m = alpha_m;
  a_form.alpha_f = alpha_f;

  L.lmbda = lambda; L.mu = mu; L.rho = rho;
  L.eta = eta;
  L.beta = beta;
  L.gamma = gamma;
  L.dt = dt;
  L.alpha_m = alpha_m;
  L.alpha_f = alpha_f;
  L.u0 = u0; L.v0 = v0; L.a0 = a0;
  L.f = f;
  L.p = p;
  L.p0 = p0;

  // Attach subdomains
  a_form.ds = right_boundary_function;
  L.ds = right_boundary_function;

  // Create projection to compute the normal strain eps_xx
  auto Vdg = std::make_shared<DG0_eps_xx::FunctionSpace>(mesh);
  DG0_eps_xx::BilinearForm a_eps(Vdg, Vdg);
  DG0_eps_xx::LinearForm L_eps(Vdg);
  L_eps.u = u;
  Function eps_xx(Vdg);

  // Create output files
  File file_u("u.pvd", "compressed");
  File file_eps("eps_xx.pvd", "compressed");

  // Start time stepping
  std::size_t step = 0;
  while (t < T)
  {
    // Update for next time step
    t += *dt;
    cout << "Time: " << t << endl;

    // Solve
    solve(a_form == L, *u, {&bc});
    solve(a_eps == L_eps, eps_xx);

    // Update velocity and acceleration
    update_a(a, *u, *a0, *v0, *u0, *beta, *dt);
    update_v(v, a, *a0, *v0, *gamma, *dt);
    *u0 = *u; *v0 = v; *a0 = a;

    // Save solutions to file
    if (step % 2 == 0)
    {
      file_u << *u;
      file_eps << eps_xx;
    }
    ++step;
  }

  return 0;
}
