// Copyright (C) 2013-2015 Anders Logg
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
// First added:  2013-06-26
// Last changed: 2016-03-02
//
// This demo program solves Poisson's equation on a domain defined by
// three overlapping and non-matching meshes. The solution is computed
// on a sequence of rotating meshes to test the multimesh
// functionality.

#include "MultiMeshPoisson.h"
#include <dolfin.h>

#include <cmath>
#include <fstream>



using namespace dolfin;
using std::make_shared;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 1.0;
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

// Compute solution for given mesh configuration
std::pair<std::shared_ptr<MultiMesh>,
          std::shared_ptr<MultiMeshFunction>> solve_poisson(double t,
                                                 double x1, double y1,
                                                 double x2, double y2,
                                                 bool plot_solution,
                                                 File& u0_file,
                                                 File& u1_file,
                                                 File& u2_file)
{
  // Create meshes
  double r = 0.5;
  auto mesh_0 = make_shared<RectangleMesh>(Point(-r, -r), Point(r, r), 16, 16);
  auto mesh_1 = make_shared<RectangleMesh>(Point(x1 - r, y1 - r), Point(x1 + r, y1 + r), 8, 8);
  auto mesh_2 = make_shared<RectangleMesh>(Point(x2 - r, y2 - r), Point(x2 + r, y2 + r), 8, 8);
  mesh_1->rotate(70*t);
  mesh_2->rotate(-70*t);

  // Build multimesh
  auto multimesh = make_shared<MultiMesh>();
  multimesh->add(mesh_0);
  multimesh->add(mesh_1);
  multimesh->add(mesh_2);
  multimesh->build();

  // Create function space
  auto V = make_shared<MultiMeshPoisson::MultiMeshFunctionSpace>(multimesh);

  // Create forms
  auto a = make_shared<MultiMeshPoisson::MultiMeshBilinearForm>(V, V);
  auto L = make_shared<MultiMeshPoisson::MultiMeshLinearForm>(V);

  // Attach coefficients
  auto f = make_shared<Source>();
  L->f = f;

  // Assemble linear system
  auto A = make_shared<Matrix>();
  auto b = make_shared<Vector>();
  assemble_multimesh(*A, *a);
  assemble_multimesh(*b, *L);

  // Apply boundary condition
  auto zero = make_shared<Constant>(0);
  auto boundary = make_shared<DirichletBoundary>();
  auto bc = make_shared<MultiMeshDirichletBC>(V, zero, boundary);
  bc->apply(*A, *b);

  // Compute solution
  auto u = make_shared<MultiMeshFunction>(V);
  solve(*A, *u->vector(), *b);

  // Save to file
  // u0_file << *u->part(0);
  // u1_file << *u->part(1);
  // u2_file << *u->part(2);

  // Plot solution (last time)
  if (plot_solution)
  {
    plot(V->multimesh());
    plot(u->part(0), "u_0");
    plot(u->part(1), "u_1");
    plot(u->part(2), "u_2");
    interactive();
  }

  return std::make_pair(multimesh, u);
}

int main(int argc, char* argv[])
{
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Sorry, this demo does not (yet) run in parallel.");
    return 0;
  }

  // Parameters
  const double T = 40.0;
  const std::size_t N = 400;
  const double dt = T / N;

  // Files for storing solution
  File u0_file("u0.pvd");
  File u1_file("u1.pvd");
  File u2_file("u2.pvd");

  // Iterate over configurations
  for (std::size_t n = 0; n < N; n++)
  {
    info("Computing solution, step %d / %d.", n + 1, N);

    // Compute coordinates for meshes
    const double t = dt*n;
    const double x1 = sin(t)*cos(2*t);
    const double y1 = cos(t)*cos(2*t);
    const double x2 = cos(t)*cos(2*t);
    const double y2 = sin(t)*cos(2*t);

    // Compute solution
    std::pair<std::shared_ptr<MultiMesh>,
              std::shared_ptr<MultiMeshFunction>> u = solve_poisson(t, x1, y1, x2, y2, n == N - 1,
                                                                    u0_file, u1_file, u2_file);

    for (int i = 0; i < 3; i++)
    {
      std::stringstream filenamestream;
      filenamestream << "ref_" << n << "_" << i << ".txt";
      const std::string filename = filenamestream.str();

      std::vector<double> values;
      u.second->part(i)->compute_vertex_values(values, *u.first->part(i));

      //   std::cout << "Writing reference file " << filename << std::endl;
      //   std::ofstream outfile(filename);
      //   outfile << std::setprecision(16);
      //   for (double d : values)
      //     outfile << d << std::endl;

      // Compare with reference value
      std::ifstream infile(filename);
      double ref_value;
      std::size_t j = 0;
      while (infile >> ref_value)
      {
        if (!near(ref_value, values[j], DOLFIN_EPS_LARGE))
        {
          std::cout << "Comparison with reference value failed: "
                    << ref_value << ", " << values[j] << " : "
                    << ref_value-values[j] << std::endl;
        }
        j++;
      }
    }
  }

  return 0;
}
