// Copyright (C) 2016 Patrick E. Farrell and Garth N. Wells
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
// This is a highly experimental demo of a geometric multigrid solver
// using PETSc. It is very likely to change.

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  { values[0] = 1.0; }
};

// Normal derivative (Neumann boundary condition)
class dUdN : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  { values[0] = sin(5*x[0]); }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { return on_boundary; }
};

int main()
{
  // Create hierarchy of meshes
  std::vector<std::shared_ptr<Mesh>> meshes
    = {std::make_shared<UnitSquareMesh>(16, 16),
       std::make_shared<UnitSquareMesh>(32, 32),
       std::make_shared<UnitSquareMesh>(64, 64)};

  // Create hierarchy of funcrion spaces
  std::vector<std::shared_ptr<const FunctionSpace>> V;
  for (auto mesh : meshes)
    V.push_back(std::make_shared<Poisson::FunctionSpace>(mesh));

  // Define boundary condition on fine grid
  auto ubc = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<DirichletBoundary>();
  auto bc = std::make_shared<DirichletBC>(V.back(), ubc, boundary);

  // Define variational forms on fine grid
  Poisson::BilinearForm a(V.back(), V.back());
  Poisson::LinearForm L(V.back());
  auto f = std::make_shared<Source>();

  // Assemble system
  auto A = std::make_shared<PETScMatrix>();
  PETScVector b;
  assemble_system(*A, b, a, L, {bc});

  // Create Krylove solver
  PETScKrylovSolver solver;
  solver.set_operator(A);

  // Set PETSc solver type
  PETScOptions::set("ksp_type", "richardson");
  PETScOptions::set("pc_type", "mg");

  // Set PETSc MG type and levels
  PETScOptions::set("pc_mg_levels", V.size());
  PETScOptions::set("pc_mg_galerkin");

  // Set smoother
  PETScOptions::set("mg_levels_ksp_type", "chebyshev");
  PETScOptions::set("mg_levels_pc_type", "jacobi");

  //Set tolerance and monitor residual
  PETScOptions::set("ksp_monitor_true_residual");
  PETScOptions::set("ksp_atol", 1.0e-12);
  PETScOptions::set("ksp_rtol", 1.0e-12);
  solver.set_from_options();

  // Create PETSc DM objects
  PETScDMCollection dm_collection(V);

  // Get fine grid DM and attach fine grid DM to solver
  solver.set_dm(dm_collection.get_dm(-1));
  solver.set_dm_active(false);

  Function u(V.back());
  solver.solve(*u.vector(), b);

  return 0;
}
