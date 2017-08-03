#include "Stokes.h"

void run_reference()
{
  info("Running reference case");

  // Read mesh and sub domain markers
  UnitSquareMesh mesh(16, 16);

  // Create function space and subspaces
  Stokes::FunctionSpace W(mesh);
  SubSpace W0(W, 0);
  SubSpace W1(W, 1);

  // Create boundary values
  Constant noslip_value(0, 0);
  InflowValue inflow_value;
  Constant outflow_value(0);

  // Create subdomains for boundary conditions
  NoslipBoundary noslip_boundary;
  InflowBoundary inflow_boundary;
  OutflowBoundary outflow_boundary;

  // No-slip boundary condition for velocity
  DirichletBC bc0(W0, noslip_value, noslip_boundary);

  // Inflow boundary condition for velocity
  DirichletBC bc1(W0, inflow_value, inflow_boundary);

  // Boundary condition for pressure at outflow
  DirichletBC bc2(W1, outflow_value, outflow_boundary);

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
}
