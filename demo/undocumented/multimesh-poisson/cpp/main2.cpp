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
// Last changed: 2015-11-30
//
// This demo program solves Poisson's equation on a domain defined by
// three overlapping and non-matching meshes. The solution is computed
// on a sequence of rotating meshes to test the multimesh
// functionality.

#include <cmath>
#include <dolfin.h>
#include "MultiMeshPoisson.h"

#include "P1.h"
#include </home/august/dev/fenics-dev/dolfin-multimesh/dolfin/geometry/dolfin_simplex_tools.h>

using namespace dolfin;

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

void evaluate_at_qr(const MultiMesh& mm,
		    const MultiMeshFunction& uh)
{
  for (std::size_t part = 0; part < mm.num_parts(); ++part)
  {
    std::cout << "\npart " << part << '\n';

    // get vertex values
    std::vector<double> vertex_values;
    uh.part(part)->compute_vertex_values(vertex_values, *mm.part(part));

    const std::vector<std::string> colors = {{ "'b'", "'g'", "'r'" }};
    std::vector<std::size_t> cells;

    // cells colliding with the cut cells
    const auto collision_map = mm.collision_map_cut_cells(part);

    // loop over cut cells
    for (const auto cell_no: mm.cut_cells(part))
    {
      // all qr on cell_no
      const auto qr = mm.quadrature_rule_cut_cell(part, cell_no);

      // loop over qr
      for (std::size_t i = 0; i < qr.second.size(); ++i)
      {
	const Point p(qr.first[2*i], qr.first[2*i+1]);
	const double uhval = (*uh.part(part))(p.x(), p.y());

	// if evaluated function big...
	if (std::abs(uhval) > 100)
	{
	  // save cell no
	  cells.push_back(cell_no);
	  const std::string color = qr.second[i] > 0 ? "'.'" : "'x'";
	  std::cout << tools::matlabplot(p,color) <<" # " << qr.second[i] << ' '
		    << /*std::setprecision(15) <<*/ uhval << " (";

	  // print nodal uh values
	  const Cell cell(*mm.part(part), cell_no);
	  for (std::size_t j = 0; j < cell.num_vertices(); ++j)
	    std::cout << cell.entities(0)[j] << ' '<<vertex_values[cell.entities(0)[j]] <<' ';
	  std::cout << ")\n";
	}
      }
    }

    // make cell numbers unique
    std::sort(cells.begin(), cells.end());
    const auto new_end = std::unique(cells.begin(), cells.end());
    cells.erase(new_end, cells.end());

    // loop over all cells with large uh values
    for (const auto cell_no: cells)
    {
      const Cell cell(*mm.part(part), cell_no);
      std::cout << tools::drawtriangle(cell);

      // compute net weight (~visible area)
      const auto qr = mm.quadrature_rule_cut_cell(part, cell_no);
      double net_weight = 0;
      for (const auto w: qr.second)
      {
	net_weight += w;
	std::cout << ' '<<w;
      }
      std::cout << " # net weight = " << net_weight << '\n';

      // also display all colliding cells
      const auto it = collision_map.find(cell_no);
      dolfin_assert(it->first == cell_no);
      for (const auto cpair: it->second)
      {
	const Cell cutting_cell(*mm.part(cpair.first), cpair.second);
	std::cout << tools::drawtriangle(cutting_cell,colors[cpair.first]);
      }
    }

  }
}

void find_max(std::size_t step,
	      const MultiMesh& multimesh,
	      const MultiMeshFunction& u,
	      File& uncut0_file, File& uncut1_file, File& uncut2_file,
	      File& cut0_file, File& cut1_file, File& cut2_file,
	      File& covered0_file, File& covered1_file, File& covered2_file)

{
  std::cout << "\tmax min step " << step <<' ' << u.vector()->max() << ' ' << u.vector()->min() << '\n';

  for (std::size_t part = 0; part < multimesh.num_parts(); ++part)
  {
    // get max on vertex values
    std::vector<double> vertex_values;
    u.part(part)->compute_vertex_values(vertex_values,
					*multimesh.part(part));
    const double maxvv = *std::max_element(vertex_values.begin(),
					   vertex_values.end());

    // get max on uncut, cut and covered
    const std::vector<std::vector<unsigned int>> cells
      = {{ multimesh.uncut_cells(part),
	   multimesh.cut_cells(part),
	   multimesh.covered_cells(part) }};
    std::vector<double> maxvals(cells.size(), 0);

    for (std::size_t k = 0; k < cells.size(); ++k)
    {
      if (cells[k].size())
      {
	// Create meshfunction using markers
	MeshFunction<std::size_t> foo(*multimesh.part(part),
				      multimesh.part(part)->topology().dim());
	foo.set_all(0); // dummy
	for (const auto cell: cells[k])
	  foo.set_value(cell, k+1);

	// Create submesh out of meshfunction
	SubMesh sm(*multimesh.part(part), foo, k+1);

	// Interpolate on submesh
	P1::FunctionSpace V(sm);
	Function usm(V);
	usm.interpolate(*u.part(part));

	// Get max values on submesh
	std::vector<double> vertex_values;
	usm.compute_vertex_values(vertex_values);
	maxvals[k] = *std::max_element(vertex_values.begin(),
				       vertex_values.end());

	// if (part == 0)
	//   if (k == 0 or k == 1) {
	//     std::cout << k <<'\n';
	//     for (const auto cell: cells[k])
	// 	std::cout << cell << ' ';
	//     std::cout << '\n';
	//   }

	// if (marker == 1 and part == 0) {
	//   for (const auto v: vertex_values)
	//     std::cout << v<<' ';
	//   std::cout << '\n';
	// }

	// save
	switch(k) {
	case 0: { // uncut
	  if (part == 0) uncut0_file << usm;
	  else if (part == 1) uncut1_file << usm;
	  else if (part == 2) uncut2_file << usm;
	  break;
	}
	case 1: { // cut
	  if (part == 0) cut0_file << usm;
	  else if (part == 1) cut1_file << usm;
	  else if (part == 2) cut2_file << usm;
	  break;
	}
	case 2: { // covered
	  if (part == 0) covered0_file << usm;
	  else if (part == 1) covered1_file << usm;
	  else if (part == 2) covered2_file << usm;
	}
	}
      }
    }

    std::cout << "\tpart " << part
	      << " step " << step
	      << " all vertices " << maxvv
	      << " uncut " << maxvals[0]
	      << " cut " << maxvals[1]
	      << " covered " << maxvals[2] << '\n';
  }

}

// Compute solution for given mesh configuration
void solve_poisson(std::size_t step,
		   double t,
                   double x1, double y1,
                   double x2, double y2,
                   bool plot_solution,
		   File& u0_file, File& u1_file, File& u2_file,
		   File& uncut0_file, File& uncut1_file, File& uncut2_file,
		   File& cut0_file, File& cut1_file, File& cut2_file,
		   File& covered0_file, File& covered1_file, File& covered2_file)
{
  // Create meshes
  double r = 0.5;
  RectangleMesh mesh_0(Point(-r, -r), Point(r, r), 16, 16);
  RectangleMesh mesh_1(Point(x1 - r, y1 - r), Point(x1 + r, y1 + r), 8, 8);
  RectangleMesh mesh_2(Point(x2 - r, y2 - r), Point(x2 + r, y2 + r), 8, 8);
  mesh_1.rotate(70*t);
  mesh_2.rotate(-70*t);



  // Build multimesh
  MultiMesh multimesh;
  multimesh.add(mesh_0);
  multimesh.add(mesh_1);
  multimesh.add(mesh_2);
  multimesh.build(); // qr generated here


  // // Create function space
  // MultiMeshPoisson::MultiMeshFunctionSpace V(multimesh);
  MultiMeshPoisson::FunctionSpace V0(mesh_0);
  MultiMeshPoisson::FunctionSpace V1(mesh_1);
  MultiMeshPoisson::FunctionSpace V2(mesh_2);

  // // Create forms
  // MultiMeshPoisson::MultiMeshBilinearForm a(V, V);
  // MultiMeshPoisson::MultiMeshLinearForm L(V);
  MultiMeshPoisson::BilinearForm a0(V0, V0);
  MultiMeshPoisson::BilinearForm a1(V1, V1);
  MultiMeshPoisson::BilinearForm a2(V2, V2);
  MultiMeshPoisson::LinearForm L0(V0);
  MultiMeshPoisson::LinearForm L1(V1);
  MultiMeshPoisson::LinearForm L2(V2);

  // Build multimesh function space
  MultiMeshFunctionSpace V;
  V.parameters("multimesh")["quadrature_order"] = 2;
  V.add(V0);
  V.add(V1);
  V.add(V2);
  V.build(); // multimesh built here, i.e. qr generation here

  // Attach coefficients
  Source f;
  // L.f = f;
  L0.f = f;
  L1.f = f;
  L2.f = f;

  // Build multimesh forms
  MultiMeshForm a(V, V);
  MultiMeshForm L(V);
  a.add(a0);
  a.add(a1);
  a.add(a2);
  L.add(L0);
  L.add(L1);
  L.add(L2);
  a.build();
  L.build();

  // Assemble linear system
  Matrix A;
  Vector b;
  // assemble_multimesh(A, a);
  // assemble_multimesh(b, L);
  MultiMeshAssembler assembler;
  assembler.assemble(A, a);
  assembler.assemble(b, L);

  // Apply boundary condition
  Constant zero(0);
  DirichletBoundary boundary;
  MultiMeshDirichletBC bc(V, zero, boundary);
  bc.apply(A, b);

  // Compute solution
  MultiMeshFunction u(V);
  solve(A, *u.vector(), b);

  // Debugging
  {
    find_max(step, multimesh, u,
	     uncut0_file, uncut1_file, uncut2_file,
	     cut0_file, cut1_file, cut2_file,
	     covered0_file, covered1_file, covered2_file);

    //evaluate_at_qr(multimesh,u);
  }

  // Save to file
  u0_file << *u.part(0);
  u1_file << *u.part(1);
  u2_file << *u.part(2);

  // Plot solution (last time)
  if (plot_solution)
  {
    plot(V.multimesh());
    plot(u.part(0), "u_0");
    plot(u.part(1), "u_1");
    plot(u.part(2), "u_2");
    interactive();
  }


}

int main(int argc, char* argv[])
{
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Sorry, this demo does not (yet) run in parallel.");
    return 0;
  }

  //set_log_level(DEBUG);

  // Parameters
  const double T = 40.0;
  const std::size_t start = 24;
  const std::size_t N = 250;
  const double dt = T / 4000;

  // Files for storing solution
  File u0_file("u0.pvd");
  File u1_file("u1.pvd");
  File u2_file("u2.pvd");

  File uncut0_file("uncut0.pvd");
  File uncut1_file("uncut1.pvd");
  File uncut2_file("uncut2.pvd");

  File cut0_file("cut0.pvd");
  File cut1_file("cut1.pvd");
  File cut2_file("cut2.pvd");

  File covered0_file("covered0.pvd");
  File covered1_file("covered1.pvd");
  File covered2_file("covered2.pvd");

  // Iterate over configurations
  for (std::size_t n = start; n < N; n++)
  {
    info("Computing solution, step %d / %d.", n, N - 1);

    // Compute coordinates for meshes
    const double t = dt*n;
    const double x1 = sin(t)*cos(2*t);
    const double y1 = cos(t)*cos(2*t);
    const double x2 = cos(t)*cos(2*t);
    const double y2 = sin(t)*cos(2*t);

    // Compute solution
    // solve_poisson(t, x1, y1, x2, y2, n == N - 1,
    //               u0_file, u1_file, u2_file);
    solve_poisson(n,
		  t, x1, y1, x2, y2, n == N - 1,
		  u0_file, u1_file, u2_file,
		  uncut0_file, uncut1_file, uncut2_file,
                  cut0_file, cut1_file, cut2_file,
		  covered0_file, covered1_file, covered2_file);

  }

  return 0;
}
