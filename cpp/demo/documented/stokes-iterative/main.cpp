// Copyright (C) 2011 Garth N. Wells
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
// First added:  2011-01-05
// Last changed: 2012-07-05
//
// This demo solves the Stokes equations using an iterative linear solver.
// Its also demonstrates how to use a precontioner matrix that is different
// from the matrix being solved.
//
// Note that the sign for the pressure has been flipped for symmetry.
//

#include "Stokes.h"
#include "StokesPreconditioner.h"
#include <dolfin.h>

using namespace dolfin;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  // Sub domain for left-hand side
  class Left : public SubDomain
  {
    bool inside(Eigen::Ref<const Eigen::VectorXd> x, bool on_boundary) const
    {
      return x[0] < DOLFIN_EPS;
    }
  };

  // Sub domain for right-hand side
  class Right : public SubDomain
  {
    bool inside(Eigen::Ref<const Eigen::VectorXd> x, bool on_boundary) const
    {
      return std::abs(1.0 - x[0]) < DOLFIN_EPS;
    }
  };

  // Sub domain for top and bottom
  class TopBottom : public SubDomain
  {
    bool inside(Eigen::Ref<const Eigen::VectorXd> x, bool on_boundary) const
    {
      return std::abs(1.0 - x[1]) < DOLFIN_EPS || std::abs(x[1]) < DOLFIN_EPS;
    }
  };

  // Function for inflow boundary condition for velocity
  class Inflow : public Expression
  {
  public:
    Inflow() : Expression({3}) {}

    void eval(Eigen::Ref<Eigen::VectorXd> values,
              Eigen::Ref<const Eigen::VectorXd> x) const
    {
      values[0] = -sin(x[1] * DOLFIN_PI);
      values[1] = 0.0;
      values[2] = 0.0;
    }
  };

  // Create mesh
  auto mesh = std::make_shared<Mesh>(
      BoxMesh::create(MPI_COMM_WORLD, {{Point(0, 0, 0), Point(1, 1, 1)}},
                      {{16, 16, 16}}, CellType::Type::hexahedron));

  // Create function space and subspaces
  auto W = std::make_shared<Stokes::FunctionSpace>(mesh);

  // Set-up infow boundary condition
  auto inflow_prfofile = std::make_shared<Inflow>();
  auto right = std::make_shared<Right>();
  auto inflow
      = std::make_shared<DirichletBC>(W->sub({0}), inflow_prfofile, right);

  // Set-up no-slip boundary condition
  std::vector<double> zero_const = {0.0, 0.0, 0.0};
  auto zero_vector = std::make_shared<Constant>(zero_const);
  auto top_bottom = std::make_shared<TopBottom>();
  auto noslip
      = std::make_shared<DirichletBC>(W->sub({0}), zero_vector, top_bottom);

  // Create forms for the Stokes problem
  auto f = std::make_shared<Constant>(zero_const);
  auto a = std::make_shared<Stokes::BilinearForm>(W, W);
  auto L = std::make_shared<Stokes::LinearForm>(W);
  L->f = f;

  // Create solution function
  Function w(W);

  // Create form for the Stokes preconditioner
  auto a_P = std::make_shared<StokesPreconditioner::BilinearForm>(W, W);

  // Assemble precondtioner system (P, b_dummy)
  auto P = std::make_shared<PETScMatrix>(mesh->mpi_comm());
  PETScVector b(mesh->mpi_comm());
  SystemAssembler s1(a_P, L, {inflow, noslip});
  s1.assemble(*P, b);

  // Assemble Stokes system (A, b)
  auto A = std::make_shared<PETScMatrix>(mesh->mpi_comm());
  SystemAssembler s2(a, L, {inflow, noslip});
  s2.assemble(*A, b);

  // Create Krylov solver with specified method and preconditioner
  PETScKrylovSolver solver(mesh->mpi_comm(), "minres", "amg");

  // Set operator (A) and precondtioner matrix (P)
  solver.set_operators(*A, *P);

  // Solve system
  solver.solve(*w.vector(), b);
  cout << "Soln norm: " << w.vector()->norm("l2") << endl;

  // Split solution
  Function u; // = w[0];
  Function p; // = w[1];

  // Save solution in VTK format
  VTKFile ufile_pvd("velocity.pvd");
  ufile_pvd.write(u);
  VTKFile pfile_pvd("pressure.pvd");
  pfile_pvd.write(p);

  MPI_Finalize();
  return 0;
}
