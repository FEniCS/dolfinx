// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include <dolfin/PDE.h>
#include <dolfin/NewtonSolver.h>
#include <dolfin/PlasticitySolver.h>
#include <dolfin/PlasticityModel.h>
#include <dolfin/PlasticityProblem.h>
#include <dolfin/Output2D.h>
#include <dolfin/Output3D.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PlasticitySolver::PlasticitySolver(Mesh& mesh, BoundaryCondition& bc, 
                  Function& f, const real dt, const real T, 
                  PlasticityModel& plasticity_model) : mesh(mesh), bc(bc), f(f), 
                  dt(dt), T(T), plasticity_model(plasticity_model)
{
  dolfin_warning("The plasticity solver is experimental.");
}
//-----------------------------------------------------------------------------
void PlasticitySolver::solve()
{
  bool elastic_tangent = false;

  // Solution function
  Function u;

  // Create PlasticityProblem
  PlasticityProblem nonlinear_problem(u, f, mesh, bc, elastic_tangent, plasticity_model);  

  // Create nonlinear solver and set parameters
  NewtonSolver nonlinear_solver;
  nonlinear_solver.set("Newton convergence criterion", "incremental");
  nonlinear_solver.set("Newton maximum iterations", 50);
  nonlinear_solver.set("Newton relative tolerance", 1e-6);
  nonlinear_solver.set("Newton absolute tolerance", 1e-10);

  // Bilinear and linear forms for continuous output of equivalent plastic strain
  BilinearForm *a_cont_equivalent_plastic_strain = 0;
  LinearForm   *L_cont_equivalent_plastic_strain = 0;

  if(mesh.topology().dim() == 2)
  {
    a_cont_equivalent_plastic_strain = new Output2D::BilinearForm;
    L_cont_equivalent_plastic_strain 
       = new Output2D::LinearForm(*(nonlinear_problem.equivalent_plastic_strain_old_function));
  }
  else if(mesh.topology().dim() == 3)
  {
    a_cont_equivalent_plastic_strain = new Output3D::BilinearForm;
    L_cont_equivalent_plastic_strain 
      = new Output3D::LinearForm(*(nonlinear_problem.equivalent_plastic_strain_old_function));
  }

  // Setup pde to project strains onto a continuous basis
  PDE pde(*a_cont_equivalent_plastic_strain, *L_cont_equivalent_plastic_strain, mesh);
  pde.set("PDE linear solver", "iterative");

  // Function to hold continuous equivalent plastic strain
  Function cont_equivalent_plastic_strain;

  // Time
  real t  = 0.0;

  // Associate source term and boundary conditions with time
  f.sync(t);
  bc.sync(t);
  
  // Solution vector
  Vector& x = u.vector();

  // File names for output
  File file1("disp.pvd");
  File file2("eq_plas_strain.pvd");

  while( t < T)
  {
    elastic_tangent = false;

    // Use elastic tangent in first time step
    if (t==0.0)
      elastic_tangent = true;

    t += dt;

    // Solve non-linear problem
    nonlinear_solver.solve(nonlinear_problem, x);

    // Update variables
    *(nonlinear_problem.plastic_strain_old_function) 
        = *(nonlinear_problem.plastic_strain_new_function);
    *(nonlinear_problem.equivalent_plastic_strain_old_function) 
        = *(nonlinear_problem.equivalent_plastic_strain_new_function);
    *(nonlinear_problem.consistent_tangent_old_function) 
        = *(nonlinear_problem.consistent_tangent_new_function);

    // Project equivalent strain onto continuous basis for postprocessing
    dolfin_log(false);
    pde.solve(cont_equivalent_plastic_strain);
    dolfin_log(true);

    // Write output to files
    file1 << u;
    file2 << cont_equivalent_plastic_strain;

    cout << "Time: t = " << t <<endl;
  }
}
//-----------------------------------------------------------------------------
void PlasticitySolver::solve(Mesh& mesh, BoundaryCondition& bc, Function& f, 
                             const real dt, const real T, 
                             PlasticityModel& plasticity_model)
{
  PlasticitySolver solver(mesh, bc, f, dt, T, plasticity_model);
  solver.solve();
}
//-----------------------------------------------------------------------------
