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
// Last changed: 2016-03-22
//
// This demo program solves Poisson's equation on a domain defined by
// three overlapping and non-matching meshes. The solution is computed
// on a sequence of rotating meshes to test the multimesh
// functionality.

#include <cmath>
#include <dolfin.h>
#include "MultiMeshPoisson.h"

#include "P1.h"
//#include </home/august/dev/fenics-dev/dolfin-multimesh/dolfin/geometry/dolfin_simplex_tools.h>
#include <dolfin/geometry/dolfin_simplex_tools.h>
#include "mmtools.h"
#include "fem.h"

using namespace dolfin;

// // Source term (right-hand side)
// class Source : public Expression
// {
//   void eval(Array<double>& values, const Array<double>& x) const
//   {
//     values[0] = 1.0;
//   }
// };


// Compute solution for given mesh configuration
void solve_poisson(std::size_t step,
		   int node_no,
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
  auto mesh_0 = std::make_shared<RectangleMesh>(Point(-r, -r), Point(r, r), 2*N , 2*N);
  auto mesh_1 = std::make_shared<RectangleMesh>(Point(x1 - r, y1 - r), Point(x1 + r, y1 + r), N, N);
  auto mesh_2 = std::make_shared<RectangleMesh>(Point(x2 - r, y2 - r), Point(x2 + r, y2 + r), N, N);
  mesh_1->rotate(70*t);
  mesh_2->rotate(-70*t);

  // {
  //   // move node number node_no a little bit in the x direction
  //   std::vector<double> coords = mesh_1->coordinates();
  //   coords[2*node_no]+=1e-16;
  //   std::cout << coords[2*node_no] << ' '<< coords[2*node_no+1] << std::endl;
  //   mesh_1->coordinates() = coords;
  // }


  // Build multimesh
  auto multimesh = std::make_shared<MultiMesh>();
  multimesh->add(mesh_0);
  multimesh->add(mesh_1);
  multimesh->add(mesh_2);
  multimesh->build(); // qr generated here

  {
    double volume = mmtools::compute_volume(*multimesh, 0);
    double area = mmtools::compute_interface_area(*multimesh, 0);
    std::cout << "volume " << volume << '\n'
	      << "area " << area << std::endl;
  }

  //mmtools::plot_normals(multimesh);
  mmtools::writemarkers(step, *multimesh);

  // Sub domain for Dirichlet boundary condition
  class DirichletBoundary : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return on_boundary;
    }
  };
  auto boundary = std::make_shared<DirichletBoundary>();

  auto u = fem::solve<MultiMeshPoisson::MultiMeshFunctionSpace,
		      MultiMeshPoisson::MultiMeshBilinearForm,
		      MultiMeshPoisson::MultiMeshLinearForm>
    (multimesh,
     boundary);

  // Debugging
  {
    // mmtools::find_max(step, *multimesh, *u,
    // 		      uncut0_file, uncut1_file, uncut2_file,
    // 		      cut0_file, cut1_file, cut2_file,
    // 		      covered0_file, covered1_file, covered2_file);

    // mmtools::evaluate_at_qr(*multimesh, *u);
  }

  // Save to file
  u0_file << *u->part(0);

  //u0_file << std::make_pair<std::shared_ptr<const Function>, double>(u->part(0), (double)step);
  // u0_file << std::make_pair<const Function*, double>(u->part(0), (double)step);
  // u1_file << std::make_pair<std::shared_ptr<const Function>, double>(u.part(1), (double)step);
  // u2_file << std::make_pair<std::shared_ptr<const Function>, double>(u.part(2), (double)step);

  // // Plot solution (last time)
  // if (plot_solution)
  // {
  //   plot(u->part(0), "u_0");
  //   plot(u->part(1), "u_1");
  //   plot(u->part(2), "u_2");
  //   interactive();
  // }

}

void manual_area_calculation_start64()
{
  // area check for start=64: we have manually measured the
  // intersection points of the three overlapping grids to be able to
  // compute the area and the length of the interfaces.

  // part 0:
  const Point cc(0.5,0.5,0);
  const Point x1( -0.403762341596028 ,0.483956950767669,0);
  const Point x3(0.454003129796834,     0.454003129215499,0);
  const Point xa(-0.403202111282917, 0.5,0);
  const Point xb(0.452396877021612,0.5, 0);
  const Point xc(0.5, 0.452396877021612,0);
  const Point b = cc-xa;
  const Point c = cc-x1;
  const Point d = xc-x1;
  // S1 = Area{x1,xa,xc,cc}
  const double S1 = 0.5*std::abs(b.cross(c)[2]) + 0.5*std::abs(c.cross(d)[2]);
  const Point e = xb-x3;
  const Point f = cc-x3;
  const Point g = xc-x3;
  // S2 = area{x3,xc,cc,xb}
  const double S2 = 0.5*std::abs(e.cross(f)[2]) + 0.5*std::abs(f.cross(g)[2]);
  const double area_part_0 = 1 - 2*S1 + S2;
  //std::cout << "\n\narea part 0:    " << area_part_0 << std::endl;

  // part 0 uncut volume (1-h)^2 plus two elements on the top and sides not cut
  const double h = 0.0625;
  const double area_uncut_part_0 = (1-h)*(1-h)+2*h*h;
  //std::cout << "area uncut part 0: " << area_uncut_part_0 << std::endl;

  // Check part 0 interface integral
  const double interface_part_0 = 2*(x1-xa).norm() + 2*(x3-x1).norm();
  //std::cout << "interface length part 0:   " << interface_part_0 << std::endl;

  // part 1 volume
  const Point y1(0.449057462336667, 0.595628463731148,0);
  const Point y2(0.600932132396578, 0.600932092071092,0);
  const Point y3(0.595628469787003,0.44905746289499,0);
  // S3 = area{x3,y1,y2,y3}
  const Point A = y1-x3;
  const Point B = y2-x3;
  const Point C = y3-x3;
  const double S3 = 0.5*std::abs(A.cross(B)[2]) + 0.5*std::abs(C.cross(B)[2]);
  const double area_part_1 = 1 - S3;// + S2;
  //std::cout << "area part 1 " << area_part_1 << std::endl;

  // part 1 uncut volume  = (1-H)*1 - 2*H*H
  const double H = 2*h;
  const double area_uncut_part_1 = 1-4*H*H;
  //std::cout << "area uncut part 1: " << area_uncut_part_1 << std::endl;

  // part 1 interface area
  //const Point y4(0.454003112031502,0.454003107715805,0);
  const Point y4=x3;
  const double interface_part_1 = (y2-y1).norm() + (y1-y4).norm();
  //std::cout << "interface length part 1:  "<< interface_part_1 << std::endl;

  // Summary
  std::cout << "\nSummary\nManually measured areas\n"
	    << "% part 0  uncut volume " << area_uncut_part_0
	    << "   total volume " << area_part_0
	    << '\n'
	    << "% part 1  uncut volume " << area_uncut_part_1
	    << "   total volume " << area_part_1
	    << '\n'
	    << "Manually measured interfaces:\n"
	    << "% part 0  total length " << interface_part_0 << '\n'
	    << "% part 1  total length " << interface_part_1 << std::endl;

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

    // here we know a priori that we have 289 vertices in mesh_0 and 81 vertices in mesh_1 and mesh_2
    // for (std::size_t node_no = 10; node_no < 11; ++node_no)
    for (std::size_t node_no = 10; node_no < 11; ++node_no)
    {
      std::cout << "\n-----------------------------------\n"
		<< "adjust node no " << node_no << std::endl;
      solve_poisson(n,
		    node_no,
		    t, x1, y1, x2, y2, false,//n == N - 1,
		    u0_file, u1_file, u2_file,
		    uncut0_file, uncut1_file, uncut2_file,
		    cut0_file, cut1_file, cut2_file,
		    covered0_file, covered1_file, covered2_file);
    }

    manual_area_calculation_start64();
  }


  return 0;
}
