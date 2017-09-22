// Copyright (C) 2017 August Johansson
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
// First added:  2013-08-24
// Last changed: 2017-08-30
//
// This demo program solves Poisson's equation on a domain defined by
// three overlapping and non-matching meshes. The solution is computed
// on a sequence of rotating meshes to test the multimesh
// functionality.

#include <cmath>
#include <fstream>
#include <dolfin.h>
#include "MultiMeshPoisson.h"
#include "MultiMeshL2Norm.h"
#include "MultiMeshH10Norm.h"

using namespace dolfin;

class Arguments
{
public:
  std::size_t N = 1;
  std::size_t Nx = 10;

  std::string print() const
  {
    std::stringstream ss;
    ss << "N" << N<<"_"
       << "Nx" << Nx;
    return ss.str();
  }

  void parse(int argc, char** argv)
  {
    std::size_t c = 1;

    if (argc > c)
    {
      this->N = atoi(argv[c]);
      c++;
      if (argc > c)
	this->Nx = atoi(argv[c]);
    }
  }
};


// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
  }
};

class ExactSolution : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1])*sin(DOLFIN_PI*x[2]);
  }
};

class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 3.0*DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1])*sin(DOLFIN_PI*x[2]);
  }
};

void build_multimesh(std::size_t N,
		     std::size_t Nx,
		     std::shared_ptr<MultiMesh> multimesh,
		     double& exact_volume)
{
  // Background mesh
  auto mesh_0 = std::make_shared<UnitCubeMesh>(Nx, Nx, Nx);
  multimesh->add(mesh_0);
  exact_volume = 1.;

  // Initialize random generator (dolfin built-in)
  dolfin::seed(1);

  for (std::size_t i = 1; i < N; ++i)
  {
    // Create domain range
    double x_a = dolfin::rand();
    double x_b = dolfin::rand();

    if (x_a > x_b)
      std::swap(x_a, x_b);
    double y_a = dolfin::rand();
    double y_b = dolfin::rand();
    if (y_a > y_b)
      std::swap(y_a, y_b);

    double z_a = dolfin::rand();
    double z_b = dolfin::rand();
    if (z_a > z_b)
      std::swap(z_a, z_b);

    // std::cout << i << ": " << x_a<<' '<<y_a<<' ' << z_a << ", " << x_b<<' '<<y_b << ' ' << z_b <<std::endl;

    // Find number of elements
    const std::size_t Nx_part = (std::size_t)std::max(std::abs(x_a - x_b)*Nx, 1.);
    const std::size_t Ny_part = (std::size_t)std::max(std::abs(y_a - y_b)*Nx, 1.);
    const std::size_t Nz_part = (std::size_t)std::max(std::abs(z_a - z_b)*Nx, 1.);

    // Create mesh
    auto mesh_i = std::make_shared<BoxMesh>(Point(x_a, y_a, z_a),
					    Point(x_b, y_b, z_b),
					    Nx_part, Ny_part, Nz_part);
    multimesh->add(mesh_i);
  }

  // Build
  multimesh->build();
}

template<class TFunctional>
double compute_error(const std::shared_ptr<MultiMeshFunctionSpace> V,
		     const std::shared_ptr<MultiMeshFunction> u_h,
		     const std::shared_ptr<Expression> u)
{
  std::cout << "Compute error" << std::endl;

  auto M = std::make_shared<MultiMeshForm>(V);

  for (std::size_t i = 0; i < V->multimesh()->num_parts(); ++i)
  {
    auto M_i = std::make_shared<TFunctional>(V->multimesh()->part(i));
    const bool deepcopy = true;
    M_i->uh = std::make_shared<const Function>(*u_h->part(i, deepcopy));
    M_i->u = u;
    M->add(M_i);
  }

  M->build();

  auto assembler = std::make_shared<MultiMeshAssembler>();
  auto m = std::make_shared<Scalar>();
  assembler->assemble(*m, *M);

  dolfin_assert(m->get_scalar_value() > 0.);
  return std::sqrt(m->get_scalar_value());
}

void solve(const std::shared_ptr<MultiMesh> multimesh)
{
  // Create function space
  auto V = std::make_shared<MultiMeshPoisson::MultiMeshFunctionSpace>(multimesh);

  // Create forms
  auto a = std::make_shared<MultiMeshPoisson::MultiMeshBilinearForm>(V, V);
  auto L = std::make_shared<MultiMeshPoisson::MultiMeshLinearForm>(V);

  // Attach source
  auto f = std::make_shared<Source>();
  L->f = f;

  // Assemble linear system
  auto A = std::make_shared<Matrix>();
  auto b = std::make_shared<Vector>();
  assemble_multimesh(*A, *a);
  assemble_multimesh(*b, *L);

  // Apply boundary condition
  auto zero = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<DirichletBoundary>();
  auto bc = std::make_shared<MultiMeshDirichletBC>(V, zero, boundary);
  bc->apply(*A, *b);

  // Compute solution
  auto uh = std::make_shared<MultiMeshFunction>(V);
  std::cout << "Solve" << std::endl;
  solve(*A, *uh->vector(), *b, "cg");

  // Compute errors
  auto exact_solution = std::make_shared<ExactSolution>();
  const double L2error = compute_error<MultiMeshL2Norm::Functional>(V, uh, exact_solution);
  const double H10error = compute_error<MultiMeshH10Norm::Functional>(V, uh, exact_solution);
  std::cout << L2error << ' ' << H10error << std::endl;
}


int main(int argc, char* argv[])
{
  Arguments args;
  args.parse(argc, argv);

  auto multimesh = std::make_shared<MultiMesh>();
  double exact_volume;
  build_multimesh(args.N, args.Nx, multimesh, exact_volume);

  const double volume = multimesh->compute_volume();
  const double volume_error = std::abs(volume - exact_volume);
  dolfin::cout << "volume error " << volume_error << dolfin::endl;

  if (volume_error > DOLFIN_EPS_LARGE)
  {
    dolfin::cout << "  large error" << dolfin::endl;
  }

  solve(multimesh);

}
