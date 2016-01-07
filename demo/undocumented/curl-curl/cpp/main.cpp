// Copyright (C) 2009-2015 Bartosz Sawicki, Stefano Zampini and Garth N. Wells
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
// Eddy currents phenomena in low conducting body can be described
// using electric vector potential and curl-curl operator:
//
//    \nabla \times \nabla \times T = - \frac{\partial B}{\partial t}
//
// Electric vector potential defined as:
//
//    \nabla \times T = J
//
// Boundary condition
//
//    J_n = 0,
//    T_t = T_w = 0, \frac{\partial T_n}{\partial n} = 0
//
// which is naturaly fulfilled for zero Dirichlet BC with Nedelec
// (edge) elements.
//
// This demo uses the auxiliary space Maxwell AMG preconditioner from
// Hypre, via PETSc. It therefore requires PETSc, configured with
// Hypre.

#include <dolfin.h>
#include "EddyCurrents.h"
#include "CurrentDensity.h"
#include "P1Space.h"

using namespace dolfin;

#if defined(HAS_PETSC) and defined(PETSC_HAVE_HYPRE)

int main()
{
  // Set PETSc as default linear algebra backend
  parameters["linear_algebra_backend"] = "PETSc";

  // Everywhere on exterior surface
  class DirichletBoundary: public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    { return on_boundary; }
  };

  // Load sphere mesh and refine uniformly
  Mesh mesh("../sphere.xml.gz");
  mesh = refine(mesh);

  // Homogeneous external magnetic field (dB/dt)
  Constant dbdt(0.0, 0.0, 1.0);

  // Dirichlet boundary condition
  auto zero = std::make_shared<Constant>(0.0, 0.0, 0.0);

  // Define function space and boundary condition
  auto V = std::make_shared<EddyCurrents::FunctionSpace>(mesh);
  auto boundary = std::make_shared<DirichletBoundary>();
  auto bc = std::make_shared<DirichletBC>(V, zero, boundary);

  // Define variational problem for T
  EddyCurrents::BilinearForm a(V, V);
  EddyCurrents::LinearForm L(V);
  L.dbdt = dbdt;

  // Solution function
  Function T(V);

  // Assemble system
  auto A = std::make_shared<PETScMatrix>();
  PETScVector b;
  assemble_system(*A, b, a, L, {bc});

  // Create PETSc Krylov solver
  PETScKrylovSolver solver("cg");

  // Get pointer to PETSc KSP object and set tolerances
  KSP ksp = solver.ksp();
  KSPSetTolerances(ksp, 1.0e-8, 1.0e-12, 1.0e10, 1000);

  // Set preconditioner to AMS from HYPRE
  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCHYPRE);
  PCHYPRESetType(pc, "ams");

  // Build discrete gradient operator and attach to preconditioner
  P1Space::FunctionSpace P1(mesh);
  auto G = DiscreteOperators::build_gradient(*V, P1);
  PCHYPRESetDiscreteGradient(pc, as_type<PETScMatrix>(*G).mat());

  // Inform preconditioner of constants in the Nedelec space
  std::vector<Function> constants(3, Function(V));
  constants[0] = Constant(1.0, 0.0, 0.0);
  constants[1] = Constant(0.0, 1.0, 0.0);
  constants[2] = Constant(0.0, 0.0, 1.0);

  PCHYPRESetEdgeConstantVectors(pc,
               as_type<PETScVector>(*constants[0].vector()).vec(),
               as_type<PETScVector>(*constants[1].vector()).vec(),
               as_type<PETScVector>(*constants[2].vector()).vec());

  // We are dealing with a zero conductivity problem (no mass term),
  // so we need to tell the preconditioner
  PCHYPRESetBetaPoissonMatrix(pc, NULL);

  // Set PETSc operators
  solver.set_operator(A);

  // Set PETSc Krylov prefix solver and set some options
  KSPSetOptionsPrefix(ksp, "eddy_");
  PETScOptions::set("eddy_ksp_monitor_true_residual");
  PETScOptions::set("eddy_ksp_view");
  KSPSetFromOptions(ksp);

  // Solve system
  KSPSolve(ksp, b.vec(), as_type<PETScVector>(*T.vector()).vec());

  // Update ghost values in solution vector
  T.vector()->apply("insert");

  // Define variational problem for J
  CurrentDensity::FunctionSpace V1(mesh);
  CurrentDensity::BilinearForm a1(V1,V1);
  CurrentDensity::LinearForm L1(V1);
  L1.T = T;

  // Solve problem using an iterative linear solver
  Function J(V1);
  Parameters p;
  p.add("linear_solver", "cg");
  p.add("symmetric", true);
  solve(a1 == L1, J, p);

  File file("current_density.pvd");
  file << J;

  // Plot solution
  plot(J);
  interactive();

  return 0;
}

#else

int main()
{
  info("This demo requires DOLFIN to be configured with PETSc (with Hypre).");
  return 0;
}

#endif
