// Copyright (C) 2014 Garth N. Wells
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
// This demo solves the equations of static linear elasticity for a
// pulley subjected to centripetal accelerations. The solver uses
// smoothed aggregation algerbaric multigri

#include <dolfin.h>
#include "Elasticity.h"

using namespace dolfin;

// Function to compute the near nullspace for elasticity - it is made
// up of the six rigid body modes
dolfin::VectorSpaceBasis build_nullspace(const dolfin::FunctionSpace& V,
                                         const GenericVector& x)
{
  // Get subspaces
  dolfin::SubSpace V0(V, 0);
  dolfin::SubSpace V1(V, 1);
  dolfin::SubSpace V2(V, 2);

  // Create vectors for nullspace basis
  std::vector<std::shared_ptr<dolfin::GenericVector>> basis(6);
  for (std::size_t i = 0; i < basis.size(); ++i)
    basis[i] = x.copy();

  // x0, x1, x2 translations
  V0.dofmap()->set(*basis[0], 1.0);
  V1.dofmap()->set(*basis[1], 1.0);
  V2.dofmap()->set(*basis[2], 1.0);

  // Rotations
  V0.set_x(*basis[3], -1.0, 1);
  V1.set_x(*basis[3],  1.0, 0);

  V0.set_x(*basis[4],  1.0, 2);
  V2.set_x(*basis[4], -1.0, 0);

  V2.set_x(*basis[5],  1.0, 1);
  V1.set_x(*basis[5], -1.0, 2);

  // Apply
  for (std::size_t i = 0; i < basis.size(); ++i)
    basis[i]->apply("add");

  // Create vector space and orthonormalize
  VectorSpaceBasis vector_space(basis);
  vector_space.orthonormalize();
  return vector_space;
}


int main()
{
  #ifdef HAS_PETSC

  // Set backend to PETSC
  parameters["linear_algebra_backend"] = "PETSc";

  // Inner surface subdomain
  class InnerSurface : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      const double r = 3.75 - x[2]*0.17;
      return (x[0]*x[0] + x[1]*x[1]) < r*r and on_boundary;
    }
  };

  // Centripetal loading function
  class CentripetalLoading : public Expression
  {
  public:

    CentripetalLoading() : Expression(3) {}
    void eval(Array<double>& values, const Array<double>& x) const
    {
      const double omega = 300.0;
      const double rho = 10.0;
      values[0] = rho*omega*omega*x[0];;
      values[1] = rho*omega*omega*x[1];
      values[2] = 0.0;
    }
  };

  // Read mesh
  Mesh mesh("../pulley.xml.gz");

  // Create right-hand side loading function
  CentripetalLoading f;

  // Set elasticity parameters
  double E  = 10.0;
  double nu = 0.3;
  Constant mu(E/(2.0*(1.0 + nu)));
  Constant lambda(E*nu/((1.0 + nu)*(1.0 - 2.0*nu)));

  // Create function space
  Elasticity::Form_a::TestSpace V(mesh);

  // Define variational problem
  Elasticity::Form_a a(V, V);
  a.mu = mu; a.lmbda = lambda;
  Elasticity::Form_L L(V);
  L.f = f;

  // Set up boundary condition on inner surface
  InnerSurface inner_surface;
  Constant zero(0.0, 0.0, 0.0);
  DirichletBC bc(V, zero, inner_surface);

  // Assemble system, applying boundary conditions and preserving
  // symmetry)
  PETScMatrix A;
  PETScVector b;
  assemble_system(A, b, a, L, bc);

  // Create solution function
  Function u(V);

  // Create near null space basis (required for smoothed aggregation
  // AMG). The solution vector is passed so that it can be copied to
  // generate compatible vectors for the nullspace.
  VectorSpaceBasis null_space = build_nullspace(V, *u.vector());

  // Create PETSc smoothed aggregation AMG preconditioner
  auto pc = std::make_shared<PETScPreconditioner>("petsc_amg");
  pc->parameters["report"] = true;
  pc->set_nullspace(null_space);

  // Set some multigrid smoother parameters
  PETScOptions::set("mg_levels_ksp_type", "chebyshev");
  PETScOptions::set("mg_levels_pc_type", "jacobi");

  // Improve estimate of eigenvalues for Chebyshev smoothing
  PETScOptions::set("gamg_est_ksp_type", "cg");
  PETScOptions::set("gamg_est_ksp_max_it", 50);

  // Create CG PETSc linear solver and turn on convergence monitor
  PETScKrylovSolver solver("cg", pc);
  solver.parameters["monitor_convergence"] = true;

  // Solve
  solver.solve(A, *(u.vector()), b);

  // Extract solution components (deep copy)
  Function ux = u[0];
  Function uy = u[1];
  Function uz = u[2];
  std::cout << "Norm (u vector): " << u.vector()->norm("l2") << std::endl;
  std::cout << "Norm (ux, uy, uz): " << ux.vector()->norm("l2") << "  "
            << uy.vector()->norm("l2") << "  "
            << uz.vector()->norm("l2") << std::endl;

  // Save solution in VTK format
  File vtk_file("elasticity.pvd", "compressed");
  vtk_file << u;

  // Extract stress and write in VTK format
  Elasticity::Form_a_s::TestSpace W(mesh);
  Elasticity::Form_a_s a_s(W, W);
  Elasticity::Form_L_s L_s(W);
  L_s.mu = mu;
  L_s.lmbda = lambda;
  L_s.disp = u;

  // Compute stress for visualisation
  Function stress(W);
  LocalSolver local_solver(std::shared_ptr<Form>(&a_s, NoDeleter()),
                           std::shared_ptr<Form>(&L_s, NoDeleter()),
                           LocalSolver::Cholesky);
  local_solver.solve_local_rhs(stress);

  File file_stress("stress.pvd");
  file_stress << stress;

  // Save colored mesh paritions in VTK format if running in parallel
  if (dolfin::MPI::size(mesh.mpi_comm()) > 1)
  {
    CellFunction<std::size_t>
      partitions(mesh, dolfin::MPI::rank(mesh.mpi_comm()));
    File file("partitions.pvd");
    file << partitions;
  }

  // Plot solution
  plot(u, "Displacement", "displacement");

  // Displace mesh and plot displaced mesh
  mesh.move(u);
  plot(mesh, "Deformed mesh");

  // Make plot windows interactive
  interactive();

  #else
  dolfin::cout << "DOLFIN must be configured with PETSc to run this demo."
               << dolfin::endl;
  #endif

 return 0;
}
