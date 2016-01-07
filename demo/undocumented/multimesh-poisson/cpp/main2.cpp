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
// Last changed: 2015-12-06
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

void writemarkers(std::size_t step,
		  const MultiMesh& mm)
{
  for (std::size_t part = 0; part < mm.num_parts(); ++part)
  {
    std::stringstream ss; ss << part;
    const std::size_t n = mm.part(part)->num_cells();
    std::vector<int> uncut(n, -1), cut(n, -1), covered(n, -1);
    for (const auto c: mm.uncut_cells(part)) uncut[c] = 0;
    for (const auto c: mm.cut_cells(part)) cut[c] = 1;
    for (const auto c: mm.covered_cells(part)) covered[c] = 2;
    tools::dolfin_write_medit_triangles("uncut"+ss.str(),*mm.part(part),step,&uncut);
    tools::dolfin_write_medit_triangles("cut"+ss.str(),*mm.part(part),step,&cut);
    tools::dolfin_write_medit_triangles("covered"+ss.str(),*mm.part(part),step,&covered);
  }
  tools::dolfin_write_medit_triangles("multimesh",mm,step);

}


double compute_volume(const MultiMesh& multimesh,
		      double exact_volume)
{
  std::cout << "\n" << __FUNCTION__<< std::endl;

  double volume = 0;
  std::vector<double> all_volumes;

  std::ofstream file("quadrature_volume.txt");
  if (!file.good()) { std::cout << "file not good\n"; exit(0); }
  file.precision(20);

  // Sum contribution from all parts
  std::cout << "Sum contributions\n";
  for (std::size_t part = 0; part < multimesh.num_parts(); part++)
  {
    std::cout << "% part " << part;
    double part_volume = 0;
    std::vector<double> status(multimesh.part(part)->num_cells(), 0);

    // Uncut cell volume given by function volume
    const auto uncut_cells = multimesh.uncut_cells(part);
    for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
    {
      const Cell cell(*multimesh.part(part), *it);
      volume += cell.volume();
      //std::cout << std::setprecision(20) << cell.volume() <<'\n';
      part_volume += cell.volume();
      status[*it] = 1;
      //file << "0 0 "<< cell.volume() << '\n';
    }

    std::cout << "\t uncut volume "<< part_volume << ' ';

    // Cut cell volume given by quadrature rule
    const auto& cut_cells = multimesh.cut_cells(part);
    for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
    {
      const auto& qr = multimesh.quadrature_rule_cut_cell(part, *it);
      for (std::size_t i = 0; i < qr.second.size(); ++i)
      {
	file << qr.first[2*i]<<' '<<qr.first[2*i+1]<<' '<<qr.second[i]<<'\n';
	volume += qr.second[i];
	part_volume += qr.second[i];
	//std::cout << qr.first[2*i]<<' '<<qr.first[2*i+1]<<'\n';
      }
      status[*it] = 2;
    }
    std::cout << "\ttotal volume " << part_volume << std::endl;

    all_volumes.push_back(part_volume);

    tools::dolfin_write_medit_triangles("status",*multimesh.part(part),part,&status);
  }
  file.close();

  return volume;
}

double compute_interface_area(const MultiMesh& multimesh,
			      double exact_area)
{
  std::cout << "\n" << __FUNCTION__ << std::endl;

  double area = 0;
  std::vector<double> all_areas;

  std::ofstream file("quadrature_interface.txt");
  if (!file.good()) { std::cout << "file not good\n"; exit(0); }
  file.precision(20);

  // Sum contribution from all parts
  std::cout << "Sum contributions\n";
  for (std::size_t part = 0; part < multimesh.num_parts(); part++)
  {
    std::cout << "% part " << part << ' ';
    double part_area = 0;
    const auto& quadrature_rules = multimesh.quadrature_rule_interface(part);

    // // Uncut cell area given by function area
    // const auto uncut_cells = multimesh.uncut_cells(part);
    // for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
    // {
    //   const Cell cell(*multimesh.part(part), *it);
    //   area += cell.area();
    // 	//std::cout << std::setprecision(20) << cell.area() <<'\n';
    //   part_area += cell.area();
    // 	status[*it] = 1;
    // 	//file << "0 0 "<< cell.area() << '\n';
    // }

    // std::cout << "\t uncut area "<< part_area << ' ';


    // Get collision map
    const auto& cmap = multimesh.collision_map_cut_cells(part);
    for (auto it = cmap.begin(); it != cmap.end(); ++it)
    {
      const unsigned int cut_cell_index = it->first;
      const auto& cutting_cells = it->second;

      // Iterate over cutting cells
      for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
      {
	// Get quadrature rule for interface part defined by
	// intersection of the cut and cutting cells
	const std::size_t k = jt - cutting_cells.begin();
	// std::cout << cut_cell_index << ' ' << k <<' ' << std::flush
	// 	    << quadrature_rules.size() << ' '
	// 	    << quadrature_rules.at(cut_cell_index).size() << "   " << std::flush;
	dolfin_assert(k < quadrature_rules.at(cut_cell_index).size());
	const auto& qr = quadrature_rules.at(cut_cell_index)[k];
	std::stringstream ss;
	for (std::size_t i = 0; i < qr.second.size(); ++i)
	{
	  file << qr.first[2*i]<<' '<<qr.first[2*i+1]<<' '<<qr.second[i]<<'\n';
	  //std::cout << qr.second[i]<<' ';
	  area += qr.second[i];
	  part_area += qr.second[i];
	  //std::cout << qr.first[2*i]<<' '<<qr.first[2*i+1]<<'\n';
	}
	//std::cout << std::endl;
      }
    }
    std::cout << "total area " << part_area << std::endl;
    all_areas.push_back(part_area);
  }
  file.close();

  return area;
}

void plot_normals(const MultiMesh& multimesh)
{
  std::cout << "\n" << __FUNCTION__ << std::endl;

  for (std::size_t part = 0; part < multimesh.num_parts(); part++)
  {
    std::cout << "% part " << part << ' ' <<std::endl;

    for (const auto cell_no: multimesh.cut_cells(part))
    {
      const auto qrmap = multimesh.quadrature_rule_interface(part).find(cell_no);
      const std::vector<quadrature_rule> qr = qrmap->second;

      const auto fnmap = multimesh.facet_normals(part).find(cell_no);
      const std::vector<std::vector<double>> normals = fnmap->second;

      //std::cout << qr.size() << ' ' << normals.size() << std::endl;
      dolfin_assert(qr.size() == normals.size());

      for (std::size_t i = 0; i < qr.size(); ++i)
      {
	for (std::size_t j = 0; j < qr[i].second.size(); ++j)
	{
	  const Point p(qr[i].first[2*j], qr[i].first[2*j+1]);
	  std::cout << tools::plot(p,"'k.'");
	  const Point n(normals[i][2*j],normals[i][2*j+1]);
	  const double d = 0.01;
	  std::cout << tools::drawarrow(p, p+d*n);
	}
	std::cout << std::endl;
      }

      // std::cout << nn.size() << ' ' << nn[0].size() << ' '  << qr.second.size() << std::endl;
      // dolfin_assert(nn.size() == qr.second.size());

      // // loop over qr
      // for (std::size_t i = 0; i < qr.second.size(); ++i)
      // {
      // 	const Point p(qr.first[2*i], qr.first[2*i+1]);
      // 	std::cout << tools::plot(p,"'k.'");
      // }


    }
  }
}


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
      std::cout << "# cell with large uh:\n";
      const Cell cell(*mm.part(part), cell_no);
      std::cout << tools::drawtriangle(cell);

      // compute net weight (~visible area)
      const auto qr = mm.quadrature_rule_cut_cell(part, cell_no);
      double net_weight = 0;
      std::cout << " # ";
      for (const auto w: qr.second)
      {
	net_weight += w;
	std::cout << ' '<<w;
      }
      std::cout << "\n# net weight = " << net_weight << '\n';

      // also display all colliding cells
      const auto it = collision_map.find(cell_no);
      dolfin_assert(it->first == cell_no);
      std::cout << "# colliding:\n";
      for (const auto cpair: it->second)
      {
	const Cell cutting_cell(*mm.part(cpair.first), cpair.second);
	std::cout << tools::drawtriangle(cutting_cell,colors[cpair.first]);
      }
    }

  }
  PPause;
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
  const std::size_t N = 8;
  const double r = 0.5;
  RectangleMesh mesh_0(Point(-r, -r), Point(r, r), 2*N, 2*N);
  RectangleMesh mesh_1(Point(x1 - r, y1 - r), Point(x1 + r, y1 + r), N, N);
  RectangleMesh mesh_2(Point(x2 - r, y2 - r), Point(x2 + r, y2 + r), N, N);
  mesh_1.rotate(70*t);
  mesh_2.rotate(-70*t);

  // Build multimesh
  MultiMesh multimesh;
  multimesh.add(mesh_0);
  multimesh.add(mesh_1);
  multimesh.add(mesh_2);
  multimesh.build(); // qr generated here

  {
    for (std::size_t p = 0; p < multimesh.num_parts(); ++p)
    {
      const auto c = multimesh.part(p)->coordinates();
      double x = 0., y = 0.;
      for (std::size_t i = 0; i < c.size(); i += 2)
      {
	x += c[i];
	y += c[i+1];
      }
      x /= c.size()/2;
      y /= c.size()/2;
      std::cout << "mesh_" << p << " center " << x << ' ' << y << std::endl;
    }
    //exit(0);
  }

  {
    double volume = compute_volume(multimesh, 0);
    double area = compute_interface_area(multimesh, 0);
    std::cout << "volume " << volume << '\n'
	      << "area " << area << std::endl;
  }

  {
    // Debug
    plot_normals(multimesh);
    writemarkers(step, multimesh);
  }


  // Create function space
  MultiMeshPoisson::MultiMeshFunctionSpace V(multimesh);

  // Create forms
  MultiMeshPoisson::MultiMeshBilinearForm a(V, V);
  MultiMeshPoisson::MultiMeshLinearForm L(V);

  // Attach coefficients
  Source f;
  L.f = f;

  // Assemble linear system
  Matrix A;
  Vector b;
  assemble_multimesh(A, a);
  assemble_multimesh(b, L);

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
  u0_file << std::make_pair<const Function*, double>(&*u.part(0), (double)step);
  u1_file << std::make_pair<const Function*, double>(&*u.part(1), (double)step);
  u2_file << std::make_pair<const Function*, double>(&*u.part(2), (double)step);

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
  const std::size_t start = 64;///228; //24
  const std::size_t N = 65;//229; //25
  const double dt = T / 400;

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

  {
    // area check for start=64
    const Point cc(0.5,0.5,0);
    const Point x1( -0.403762341596028 ,0.483956950767669,0);
    const Point x3(0.454003129796834,     0.454003129215499,0);
    const Point xa(-0.403202111282917, 0.5,0);
    const Point xb(0.452396877021612,0.5, 0);
    const Point xc(0.5, 0.452396877021612,0);
    const Point b = cc-xa;
    const Point c = cc-x1;
    const Point d = xc-x1;
    const double S1 = 0.5*std::abs(b.cross(c)[2]) + 0.5*std::abs(c.cross(d)[2]);
    const Point e = xb-x3;
    const Point f = cc-x3;
    const Point g = xc-x3;
    const double S2 = 0.5*std::abs(e.cross(f)[2]) + 0.5*std::abs(f.cross(g)[2]);
    const double area_part_0 = 1 - 2*S1 + S2;
    std::cout << "\n\narea part 0:    " << area_part_0 << std::endl;

    // uncut volume (1-h)^2 plus two elements on the top and sides not cut
    const double h = 0.0625;
    std::cout << "area uncut part 0: " << (1-h)*(1-h)+2*h*h << std::endl;

    // Check interface integral
    double interface_part_0 = 2*(x1-xa).norm() + 2*(x3-x1).norm();
    std::cout << "interface length part 0:   " << interface_part_0 << std::endl;
  }


  return 0;
}
