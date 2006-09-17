// Copyright (C) 2006 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006
// Last changed: 2006-05-04

//#ifdef HAVE_PETSC_H

#include <dolfin/dolfin_log.h>
#include <dolfin/FEM.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/GMRES.h>
#include <dolfin/LU.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/Mesh.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/Function.h>
#include <dolfin/TimeDependentPDE.h>
#include <dolfin/Parametrized.h>
#include <dolfin/TimeStepper.h>

using namespace dolfin;

TimeDependentPDE::TimeDependentPDE(BilinearForm& a, LinearForm& L, Mesh& mesh, 
  BoundaryCondition& bc, int N, real k, real T) : GenericPDE(), x(0), k(k),
						  _a(&a), _Lf(&L),
						  _mesh(&mesh), _bc(&bc),
						  N(N), t(0), T(T)
{
  x = new Vector(N);
  dotx = new Vector(N);

  for(unsigned int i = 0; i < this->L().num_functions; i++)
  {
    Function* f = this->L().function(i);
    f->sync(t);
  }
}
//-----------------------------------------------------------------------------
TimeDependentPDE::~TimeDependentPDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint TimeDependentPDE::solve(Function& U)
{
  dolfin_assert(_a);
  dolfin_assert(_Lf);
  dolfin_assert(_mesh);

  init(U);

  // Make sure u is a discrete function associated with the trial space
//   u.init(*_mesh, _a->trial());
//   x = &(u.vector());
//   x = new Vector(N);
//   dotx = new Vector(N);

  //int N = FEM::size(*_mesh, _Lf->test());

  cout << "N: " << N << endl;
  cout << "x size: " << x->size() << endl;


  // Initialize ODE (requires x)
  ode = new TimeDependentODE(*this, N, T);
  ts = new TimeStepper(*ode);

  // Write a message
  dolfin_info("Solving time dependent PDE.");

//   File  solutionfile("solution.pvd");

//   solutionfile << U;

//   int counter = 0;

  save(U, t);

  // Start time-stepping
  while(t < T) {
//     cout << "t: " << t << endl;
    
    preparestep();
    t = ts->step();



//     if((counter % 333 * 2) == 0)
//     {
//       solutionfile << U;
//     }

//     counter++;
    save(U, t);
  }

  return 0;
}
//-----------------------------------------------------------------------------
void TimeDependentPDE::fu(const Vector& x, Vector& dotx, real t)
{
}
//-----------------------------------------------------------------------------
dolfin::uint TimeDependentPDE::elementdim()
{
  dolfin_assert(_a);
  return _a->trial().elementdim();
}
//-----------------------------------------------------------------------------
BilinearForm& TimeDependentPDE::a()
{
  dolfin_assert(_a);
  return *_a;
}
//-----------------------------------------------------------------------------
LinearForm& TimeDependentPDE::L()
{
  dolfin_assert(_Lf);
  return *_Lf;
}
//-----------------------------------------------------------------------------
Mesh& TimeDependentPDE::mesh()
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
BoundaryCondition& TimeDependentPDE::bc()
{
  dolfin_assert(_bc);
  return *_bc;
}
//-----------------------------------------------------------------------------
void TimeDependentPDE::init(Function& U)
{
}
//-----------------------------------------------------------------------------
void TimeDependentPDE::save(Function& U, real t)
{
}
void TimeDependentPDE::preparestep()
{
}
//-----------------------------------------------------------------------------
void TimeDependentPDE::prepareiteration()
{
}
//-----------------------------------------------------------------------------
TimeDependentODE::TimeDependentODE(TimeDependentPDE& pde, int N, real T) :
  ODE(N, T), pde(&pde)
{
}
//-----------------------------------------------------------------------------
void TimeDependentODE::u0(uBlasVector& u)
{
  // FIXME: ODE solver interface has changed
  //dolfin_error("Not implemented.");
  u.copy(*(pde->x), 0, 0, u.size());
}
//-----------------------------------------------------------------------------
void TimeDependentODE::f(const uBlasVector& u, real t, uBlasVector& y)
{
  pde->x->copy(u, 0, 0, u.size());

  pde->prepareiteration();

  pde->fu(*(pde->x), *(pde->dotx), t);

  y.copy(*(pde->dotx), 0, 0, u.size());
}
//-----------------------------------------------------------------------------
bool TimeDependentODE::update(const uBlasVector& u, real t, bool end)
{
  return true;
}
//-----------------------------------------------------------------------------
real TimeDependentODE::timestep(real t, real k0) const
{
  return pde->k;
}
//-----------------------------------------------------------------------------

//#endif
