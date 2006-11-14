// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include "dolfin/PlasticitySolver.h"
#include "dolfin/Plas2D.h"
#include "dolfin/Strain2D.h"
#include "dolfin/Tangent2D.h"
#include "dolfin/Output2D.h"
#include "dolfin/p_strain2D.h"
#include "dolfin/ep_strain2D.h"
#include "dolfin/Plas3D.h"
#include "dolfin/Strain3D.h"
#include "dolfin/Tangent3D.h"
#include "dolfin/Output3D.h"
#include "dolfin/p_strain3D.h"
#include "dolfin/ep_strain3D.h"
#include "dolfin/PlasticityModel.h"
#include "dolfin/PlasticityProblem.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PlasticitySolver::PlasticitySolver(Mesh& mesh,
         BoundaryCondition& bc, Function& f, real E, real nu,
         real dt, real T, PlasticityModel& plas)
  : mesh(mesh), bc(bc), f(f), E(E), nu(nu), dt(dt), T(T), plas(plas)
{
  dolfin_warning("The plasticity solver is experimental.");
}
//-----------------------------------------------------------------------------
void PlasticitySolver::solve()
{
  //  stiffness parameters
  double lam = nu*E/((1+nu)*(1-2*nu));
  double mu = E/(2*(1+nu));

  // elastic tangent
  uBlasDenseMatrix D = C_m(lam, mu);

  bool elastic_tangent = false;

  // solution function
  Function u;

  // assemple matrix for strain computation, since it is constant it can be pre computed
  BilinearForm *a_strain = 0;
  if(mesh.topology().dim() == 2)
    a_strain = new Strain2D::BilinearForm;
  else if(mesh.topology().dim() == 3)
    a_strain = new Strain3D::BilinearForm;
  Matrix A_strain;
  FEM::assemble(*a_strain, A_strain, mesh);

  // create object of type PlasticityProblem
  PlasticityProblem nonlinear_problem(u, f, A_strain, mesh, bc, elastic_tangent, plas, D);  

  // Create nonlinear solver and set parameters
  NewtonSolver nonlinear_solver;

  nonlinear_solver.set("Newton convergence criterion", "incremental");
  nonlinear_solver.set("Newton maximum iterations", 50);
  nonlinear_solver.set("Newton relative tolerance", 1e-6);
  nonlinear_solver.set("Newton absolute tolerance", 1e-10);

  // bilinear and linear forms for continuous output of equivalent plastic strain
  BilinearForm *eq_strain_a = 0;
  LinearForm *eq_strain_L = 0;
  if(mesh.topology().dim() == 2)
  {
    eq_strain_a = new Output2D::BilinearForm;
    eq_strain_L = new Output2D::LinearForm(*nonlinear_problem.eq_strain_old);
  }
  else if(mesh.topology().dim() == 3)
  {
    eq_strain_a = new Output3D::BilinearForm;
    eq_strain_L = new Output3D::LinearForm(*nonlinear_problem.eq_strain_old);
  }

  // setup pde to project strains onto a continuous basis
  PDE pde(*eq_strain_a, *eq_strain_L, mesh);
   pde.set("PDE linear solver", "iterative");

  // Function to hold continuous equivalent plastic strain
  Function eq_strain;

  // time
  double t  = 0.0;

  f.sync(t);      // Associate time with source term
  bc.sync(t);     // Associate time with boundary conditions
  
  // solution vector
  Vector& x = u.vector();

  // file names for output
  File file1("disp.pvd");
  File file2("eq_plas_strain.pvd");

  while( t < T)
  {
    elastic_tangent = false;

    // use elastic tangent in first time step
    if (t==0.0)
      elastic_tangent = true;

    t += dt;

    // solve non-linear problem
    nonlinear_solver.solve(nonlinear_problem, x);

    // update variables
    *nonlinear_problem.p_strain_old = *nonlinear_problem.p_strain_new;
    *nonlinear_problem.eq_strain_old = *nonlinear_problem.eq_strain_new;
    *nonlinear_problem.tangent_old = *nonlinear_problem.tangent_new;

    dolfin_log(false);
    // project strain onto continuous basis
    pde.solve(eq_strain);
    dolfin_log(true);

    // write output to files
    file1 << u;
    file2 << eq_strain;

    cout << "Time: t = " << t <<endl;
  }
}
//-----------------------------------------------------------------------------
void PlasticitySolver::solve(Mesh& mesh,
         BoundaryCondition& bc, Function& f, real E, real nu,
         real dt, real T, PlasticityModel& plas)
{
  PlasticitySolver solver(mesh, bc, f, E, nu, dt, T, plas);
  solver.solve();
}
//-----------------------------------------------------------------------------

// constitutive relation (elastic tangent)
uBlasDenseMatrix PlasticitySolver::C_m(double &lam, double &mu)
{
  uBlasDenseMatrix B(6,6);
  B.clear();

  B(0,0)=lam+2*mu, B(1,1)=lam+2*mu, B(2,2)=lam+2*mu;
  B(3,3)=mu, B(4,4)=mu, B(5,5)=mu;
  B(0,1)=lam, B(0,2)=lam, B(1,0)=lam;
  B(1,2)=lam, B(2,0)=lam, B(2,1)=lam;

  return B;
}
