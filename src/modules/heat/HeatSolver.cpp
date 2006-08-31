// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005-2006.
//
// First added:  2005-11-01
// Last changed: 2006-05-07

#include <dolfin/HeatSolver.h>
#include <dolfin/Heat.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
HeatSolver::HeatSolver(Mesh& mesh, Function& f, BoundaryCondition& bc, real& T)
  : Solver(), mesh(mesh), f(f), bc(bc), u(x, mesh, element),
    L(u, f), fevals(0),
    ode(0), ts(0), T(T)
{
  N = FEM::size(mesh, element);

  x.init(N);
  x = 0.0;
  dotu.init(N);

  Matrix M;

  // Assemble mass matrix
  FEM::assemble(a, M, mesh);

  // Lump mass matrix
  FEM::lump(M, m);

  // Dummy matrix needed for boundary condition function
  Dummy.init(N, N);
  for(uint i = 0; i < N; i++)
  {
    Dummy(i, i) = 0.0;
  }

  ode = new HeatODE(*this);
  ts = new TimeStepper(*ode);
}
//-----------------------------------------------------------------------------
void HeatSolver::solve()
{
  File file(get("solution file name"));
  // Save function to file
  file << u;

  real t = 0.0;
  while(t < T)
  {
    t = ts->step();
    // cout << "t: " << t << endl;
    // Save function to file
    dolfin_log(false);
    file << u;
    dolfin_log(true);
  }
  cout << "total fevals: " << fevals << endl;
}
//-----------------------------------------------------------------------------
void HeatSolver::fu()
{
  dolfin_log(false);
  FEM::assemble(L, dotu, mesh);
  FEM::applyBC(Dummy, dotu, mesh, element, bc);
  //VecPointwiseDivide(dotu.vec(), dotu.vec(), m.vec());
  fevals++;
  dolfin_log(true);
}
//-----------------------------------------------------------------------------
void HeatSolver::solve(Mesh& mesh, Function& f, BoundaryCondition& bc, real& T)
{
  HeatSolver solver(mesh, f, bc, T);
  solver.solve();
}
//-----------------------------------------------------------------------------
HeatODE::HeatODE(HeatSolver& solver) :
  ODE(solver.N, solver.T), solver(solver)
{
}
//-----------------------------------------------------------------------------
void HeatODE::u0(uBlasVector& u)
{
  u.copy(solver.x);
}
//-----------------------------------------------------------------------------
void HeatODE::f(const uBlasVector& u, real t, uBlasVector& y)
{
  // Copy values from ODE array
  solver.x.copy(u);

  // Compute solver RHS (puts result in Vector variables)
  solver.fu();

  // Copy values into ODE array
  y.copy(solver.dotu);
}
//-----------------------------------------------------------------------------
bool HeatODE::update(const uBlasVector& u, real t, bool end)
{
  return true;
}
//-----------------------------------------------------------------------------
