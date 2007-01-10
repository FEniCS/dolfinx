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
  : Solver(), element(new Heat::LinearForm::TestElement()),
    mesh(mesh), f(f), bc(bc), u(x, mesh, *element),
    fevals(0),
    ode(0), ts(0), T(T)
{
  a = new Heat::BilinearForm();
  L = new Heat::LinearForm(u, f);

  N = FEM::size(mesh, *element);

  x.init(N);
  x = 0.0;
  dotu.init(N);

  Matrix M;

  // Assemble mass matrix
  FEM::assemble(*a, M, mesh);

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

  int counter = 0;
  real k = T / 100;
  real lastsample = 0.0;

  real t = 0.0;
  while(t < T)
  {
    t = ts->step();
    // cout << "t: " << t << endl;
    // Save function to file
    dolfin_log(false);
    while(lastsample + k < t)
    {
      lastsample = std::min(t, lastsample + k);
      file << u;
    }
    dolfin_log(true);

    counter++;
  }
  cout << "total fevals: " << fevals << endl;
}
//-----------------------------------------------------------------------------
void HeatSolver::fu()
{
  dolfin_log(false);
  FEM::assemble(*L, dotu, mesh);
  FEM::applyBC(Dummy, dotu, mesh, *element, bc);
  dotu.div(m);
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
  u.copy(solver.x, 0, 0, u.size());
}
//-----------------------------------------------------------------------------
void HeatODE::f(const uBlasVector& u, real t, uBlasVector& y)
{
  // Copy values from ODE array
  solver.x.copy(u, 0, 0, u.size());

  // Compute solver RHS (puts result in Vector variables)
  solver.fu();

  // Copy values into ODE array
  y.copy(solver.dotu, 0, 0, y.size());
}
//-----------------------------------------------------------------------------
bool HeatODE::update(const uBlasVector& u, real t, bool end)
{
  return true;
}
//-----------------------------------------------------------------------------

