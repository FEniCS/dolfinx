// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-01
// Last changed: 2005-11-01

#include <dolfin/Heat.h>
#include <dolfin/HeatSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
HeatSolver::HeatSolver(Mesh& mesh, Function& f, BoundaryCondition& bc, real& T)
  : Solver(), mesh(mesh), f(f), bc(bc), u(x, mesh, element),
    L(u, f), N(mesh.noNodes()),
    ode(0), ts(0), T(T)
{
  uint N = mesh.noNodes();

  x.init(N);
  x = 0.0;
  dotu.init(N);

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
  File file("heat.pvd");
  // Save function to file
  file << u;

  real t = 0.0;
  while(t < T)
  {
    t = ts->step();
    cout << "t: " << t << endl;
    // Save function to file
    file << u;
  }
}
//-----------------------------------------------------------------------------
void HeatSolver::fu()
{
  FEM::assemble(L, dotu, mesh);
  FEM::applyBC(Dummy, dotu, mesh, element, bc);
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
real HeatODE::u0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
void HeatODE::f(const real u[], real t, real y[])
{
  // Copy values from ODE array
  fromArray(u, solver.x, 0, solver.N);

  // Compute solver RHS (puts result in Vector variables)
  solver.fu();

  // Copy values into ODE array
  toArray(y, solver.dotu, 0, solver.N);
}
//-----------------------------------------------------------------------------
bool HeatODE::update(const real u[], real t, bool end)
{
  return true;
}
//-----------------------------------------------------------------------------
void HeatODE::fromArray(const real u[], Vector& x, uint offset,
				     uint size)
{
  // Workaround to interface Vector and arrays

  real* vals = 0;
  vals = x.array();
  for(uint i = 0; i < size; i++)
  {
    vals[i] = u[i + offset];
  }
  x.restore(vals);
}
//-----------------------------------------------------------------------------
void HeatODE::toArray(real y[], Vector& x, uint offset, uint size)
{
  // Workaround to interface Vector and arrays

  real* vals = 0;
  vals = x.array();
  for(uint i = 0; i < size; i++)
  {
    y[offset + i] = vals[i];
  }
  x.restore(vals);
}
//-----------------------------------------------------------------------------
