// Copyright (C) 2014 August Johansson and Anders Logg
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
// First added:  2014-03-10
// Last changed: 2016-11-14
//


//#define CGAL_HEADER_ONLY 1
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/math/basic.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/generation/UnitSquareMesh.h>

// We need to use epeck here. Qoutient<MP_FLOAT> as number type gives overflow
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include <CGAL/Triangle_2.h>
#include <CGAL/intersection_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Polygon_set_2.h>

typedef CGAL::Epeck ExactKernel;
typedef ExactKernel::FT FT;
typedef ExactKernel::Point_2                      Point_2;
typedef ExactKernel::Triangle_2                   Triangle_2;
typedef ExactKernel::Line_2                       Line_2;
typedef CGAL::Polygon_2<ExactKernel>              Polygon_2;
typedef Polygon_2::Vertex_const_iterator          Vertex_const_iterator;
typedef CGAL::Polygon_with_holes_2<ExactKernel>   Polygon_with_holes_2;
typedef Polygon_with_holes_2::Hole_const_iterator Hole_const_iterator;
typedef CGAL::Polygon_set_2<ExactKernel>          Polygon_set_2;

// FIXME
#include </home/august/dolfin_simplex_tools.h>


#define MULTIMESH_DEBUG_OUTPUT 0

using namespace dolfin;

// FIXME: Add CGAL reference implementation

//------------------------------------------------------------------------------
inline double compute_area_using_quadrature(const MultiMesh& multimesh)
{
  //std::cout  << __FUNCTION__ << std::endl;

  double area = 0;
  std::vector<double> all_areas;

  std::ofstream file("quadrature_interface.txt");
  if (!file.good()) { std::cout << "file not good"<<std::endl; exit(0); }
  file.precision(20);

  // Sum contribution from all parts
  // std::cout << "Sum contributions"<<std::endl;
  for (std::size_t part = 0; part < multimesh.num_parts(); part++)
  {
    //std::cout << "% part " << part << '\n';
    double part_area = 0;
    const auto& quadrature_rules = multimesh.quadrature_rule_interface(part);

    // // Uncut cell area given by function area
    // const auto uncut_cells = multimesh.uncut_cells(part);
    // for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
    // {
    //   const Cell cell(*multimesh.part(part), *it);
    //   area += cell.area();
    // 	//std::cout << std::setprecision(20) << cell.area() <<std::endl;
    //   part_area += cell.area();
    // 	status[*it] = 1;
    // 	//file << "0 0 "<< cell.area() << std::endl;
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
	  file << qr.first[2*i]<<' '<<qr.first[2*i+1]<<' '<<qr.second[i]<<std::endl;
	  //std::cout << qr.second[i]<<' ';
	  area += qr.second[i];
	  part_area += qr.second[i];
	  // std::cout << qr.first[2*i]<<' '<<qr.first[2*i+1]<<std::endl;
	}
	//tools::cout_qr(qr);
	//std::cout << std::endl;
      }
    }
    //std::cout << "% total area " << part_area << std::endl;
    all_areas.push_back(part_area);
    // {char apa; std::cout << "paused at " << __FUNCTION__<<' '<<__LINE__<<std::endl; std::cin>>apa;}
  }
  file.close();

  return area;
}
//-----------------------------------------------------------------------------
std::shared_ptr<MultiMesh> test_meshes_on_diagonal(std::size_t Nx,
						   double width,
						   double offset,
						   std::size_t max_num_parts = 99999)
{

  // // Test squares in diagonal on background unit square
  // //const std::size_t m = 2, n = 2;//5, n = 5;
  // const std::size_t n = m;
  // const double h = 0.1111;
  // const double s = 0.3;
  // if (h >= s) { std::cout << "h must be less than s\n"; exit(1); }
  // auto usm = std::make_shared<UnitSquareMesh>(m, n);
  // auto mm = std::make_shared<MultiMesh>();
  // mm->add(usm);
  // std::vector<Mesh> meshes;
  // double exact_area = 4*s;
  // std::size_t N = 1;
  // while (N*h+s < 1)
  // {
  //   std::cout << "rectangle mesh points [" << N*h << "," << N*h+s << "] x [" << N*h << "," << N*h+s << "]" << std::endl;
  //   std::shared_ptr<Mesh> rm(new RectangleMesh(Point(N*h, N*h), Point(N*h+s, N*h+s), m, n));
  //   mm->add(rm);
  //   if (N > 1)
  //     exact_area += 2*s + 2*h;
  //   N++;
  // }
  // mm->build();
  // tools::dolfin_write_medit_triangles("multimesh", *mm, N);

  // const double area = compute_interface_area(*mm, exact_area);
  // const double error = std::abs(area - exact_area);
  // std::cout << std::setprecision(15)
  // 	    << "N = " << N << '\n'
  // 	    << "area = " << area << " (exact_area = " << exact_area <<")\n"
  // 	    << "error = " << error << '\n';
  // //CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_area, area, DOLFIN_EPS_LARGE);
  // //dolfin_assert(error<DOLFIN_EPS_LARGE);

  /*
    # Number of elements
    Nx = 1

    # Background mesh
    mesh_0 = UnitSquareMesh(Nx, Nx)

    # Create multimesh
    multimesh = MultiMesh()
    multimesh.add(mesh_0)

    # Mesh width (must be less than 1)
    width = 0.3
    assert width < 1

    # Mesh placement (must be less than the width)
    offset = 0.1111
    assert offset < width

    # Exact area for there are one mesh on top
    exact_area = 4*width

    num_parts = multimesh.num_parts()
    while num_parts*width + offset < 1:
        a = num_parts*offset
        b = a + width
        mesh_top = RectangleMesh(Point(a,a), Point(b,b), Nx, Nx)
        multimesh.add(mesh_top)
        if num_parts > 1:
            exact_area += 2*offset + 2*width
        num_parts = multimesh.num_parts()

    multimesh.build()

    area = compute_area_using_quadrature(multimesh)
    print("area ", area)
    print("exact area ", exact_area)
    print("DOLFIN_EPS_LARGE", DOLFIN_EPS_LARGE)
    assert abs(area - exact_area) < DOLFIN_EPS_LARGE
  */

  // Number of elements
  // std::size_t Nx = m;

  // Background mesh
  auto mesh_0 = std::make_shared<UnitSquareMesh>(Nx, Nx);

  // Create multimesh
  auto multimesh = std::make_shared<MultiMesh>();
  multimesh->add(mesh_0);

  // Mesh width (must be less than 1)
  // const double width = 0.6;//1./DOLFIN_PI;
  dolfin_assert(width < 1);

  // Mesh placement (must be less than the width)
  // const double offset = 0.1;//width-1e-5;//DOLFIN_PI / 300.;
  dolfin_assert(offset < width);

  // Exact area for there are one mesh on top
  // double exact_area = 4*width;

  std::size_t num_parts = multimesh->num_parts();

  while (num_parts*offset + width < 1 and
	 num_parts < max_num_parts)
  {
    //std::cout << num_parts << ' '<<num_parts*offset+width << std::endl;
    const double a = num_parts*offset;
    const double b = a + width;
    auto mesh_top = std::make_shared<RectangleMesh>(Point(a,a), Point(b,b), Nx, Nx);
    multimesh->add(mesh_top);
    num_parts = multimesh->num_parts();
  }

  const double exact_area = (num_parts >= 2) ? 4*width + (num_parts-2)*(2*offset + 2*width) : 0;

  std::cout << "num parts before build " << num_parts << std::endl;
  multimesh->build();
  std::cout << "num parts " << multimesh->num_parts() << std::endl;

  std::stringstream ss;
  ss << "multimesh_"<<Nx<<"_"<<width<<"_"<<offset<<"_"<<max_num_parts;
  tools::dolfin_write_medit_triangles(ss.str(), *multimesh, Nx);
  //tools::writematlab(ss.str(), *multimesh, false, Nx);
  //tools::writemarkers("multimesh",*multimesh,Nx);

  const double area = compute_area_using_quadrature(*multimesh);
  const double error = std::abs(area-exact_area);

  // FIXME: This can be done better
  const auto& cmap = multimesh->collision_map_cut_cells(0);
  std::size_t n_cutting_cells = 0;
  for (auto it = cmap.begin(); it != cmap.end(); ++it)
    n_cutting_cells += it->second.size();

  const double tol = std::max(DOLFIN_EPS_LARGE,
			      multimesh->num_parts()*n_cutting_cells*DOLFIN_EPS);

  std::cout << "area " << area <<std::endl;
  std::cout << "exact area " << exact_area << std::endl;
  std::cout << "error " << error << std::endl;
  std::cout << "tol " << tol << std::endl;

  dolfin_assert(error < tol);

  return multimesh;

}

//-----------------------------------------------------------------------------
template<class T>
inline std::vector<T> linspace(T a, T b, int n)
{
  std::vector<T> x(n);
  x[0]=a;
  x[n-1]=b;
  const T h=(b-a)/(n-1);
  for (int i=1; i<n-1; ++i) {
    x[i]=a+h*i; // FIXME: can probably do this more stable
  }
  return x;
}

//-----------------------------------------------------------------------------
std::vector<double> logspace(double a, double b, int n)
{
  std::vector<double> linsp=linspace(a, b, n);
  std::vector<double> logsp(n);
  for (int i=0; i<n; ++i)
    logsp[i] = std::pow(10, linsp[i]);
  return logsp;
}

//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{


  for (std::size_t Nx = 1; Nx < 2; ++Nx)
    for (std::size_t width_factor = 1; width_factor < 100; ++width_factor)
    {

      const double width = 3*width_factor/(100*DOLFIN_PI);

      // Generate offsets
      std::vector<double> offsets = {0.1*width};

      //for (std::size_t offset_factor = 1; offset_factor < 100; ++offset_factor)
      for (const double offset: offsets)
      {
	//const double offset = offset_factor*DOLFIN_PI / (100*3.2);
	if (offset < width)
	{
	  std::cout << "\n"
		    << "Nx " << Nx << " width " << width << " offset " << offset << std::endl;
	  std::shared_ptr<MultiMesh> m = test_meshes_on_diagonal(Nx, width, offset, 20);
	}
      }
    }
}
