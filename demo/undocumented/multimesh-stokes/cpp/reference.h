#include "Stokes.h"

using namespace dolfin;

// Function for no-slip boundary condition for velocity
class Noslip : public Expression
{
public:

  Noslip() : Expression(2) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 0.0;
    values[1] = 0.0;
  }

};

// Function for inflow boundary condition for velocity
class Inflow : public Expression
{
public:

  Inflow() : Expression(2) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(x[1]*DOLFIN_PI);
    values[1] = 0.0;
  }

};

// Subdomain for no-slip boundary condition
class NoslipBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && (near(x[1], 0.0) || near(x[1], 1.0));
  }
};

// Subdomain for inflow boundary condition
class InflowBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && near(x[0], 0.0);
  }
};

// Subdomain for outflow boundary condition
class OutflowBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && near(x[0], 1.0);
  }
};

void run_reference()
{
  info("Running reference case");

  // Read mesh and sub domain markers
  UnitSquareMesh mesh(16, 16);

  // Create function space and subspaces
  Stokes::FunctionSpace W(mesh);
  SubSpace W0(W, 0);
  SubSpace W1(W, 1);

  // Create functions for boundary conditions
  Noslip noslip;
  Inflow inflow;
  Constant zero(0);

  // Create subdomains for boundary conditions
  NoslipBoundary noslip_boundary;
  InflowBoundary inflow_boundary;
  OutflowBoundary outflow_boundary;

  // No-slip boundary condition for velocity
  DirichletBC bc0(W0, noslip, noslip_boundary);

  // Inflow boundary condition for velocity
  DirichletBC bc1(W0, inflow, inflow_boundary);

  // Boundary condition for pressure at outflow
  DirichletBC bc2(W1, zero, outflow_boundary);

  // Collect boundary conditions
  std::vector<const DirichletBC*> bcs;
  bcs.push_back(&bc0); bcs.push_back(&bc1); bcs.push_back(&bc2);

  // Define variational problem
  Constant f(0.0, 0.0);
  Stokes::BilinearForm a(W, W);
  Stokes::LinearForm L(W);
  L.f = f;

  // Compute solution
  Function w(W);
  solve(a == L, w, bcs);
  Function u = w[0];
  Function p = w[1];

  // Plot solution
  plot(u);
  plot(p);
  interactive();
}
