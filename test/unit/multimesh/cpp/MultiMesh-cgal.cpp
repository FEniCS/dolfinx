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
// Last changed: 2015-06-08
//
// Unit tests for MultiMesh

#include <dolfin/common/unittest.h>
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/math/basic.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/generation/UnitSquareMesh.h>

#include <CGAL/Triangle_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/intersection_2.h>

#define MULTIMESH_DEBUG_OUTPUT 1

using namespace dolfin;

class MultiMeshesCgal : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MultiMeshesCgal);
  CPPUNIT_TEST(test_multiple_meshes_with_rotation);
  CPPUNIT_TEST_SUITE_END();

public:

  //------------------------------------------------------------------------------
  double rotate(double x, double y, double cx, double cy, double w,
		double& xr, double& yr)
  {
    // std::cout << "rotate:\n"
    // 	      << "\t"
    // 	      << "plot("<<x<<','<<y<<",'b.');plot("<<cx<<','<<cy<<",'o');";

    const double v = w*DOLFIN_PI/180.;
    const double dx = x-cx;
    const double dy = y-cy;
    xr = cx + dx*cos(v) - dy*sin(v);
    yr = cy + dx*sin(v) + dy*cos(v);
    //std::cout << "plot("<<xr<<','<<yr<<",'r.');"<<std::endl;
  }
  //------------------------------------------------------------------------------
  bool rotation_inside(double x,double y, double cx, double cy, double w,
		       double& xr, double& yr)
  {
    rotate(x,y,cx,cy,w, xr,yr);
    if (xr>0 and xr<1 and yr>0 and yr<1) return true;
    else return false;
  }
  //------------------------------------------------------------------------------
  double compute_volume(const MultiMesh& multimesh) const
  {
    double volume = 0;

    // Sum contribution from all parts
    for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    {
      // Uncut cell volume given by function volume
      const std::vector<unsigned int>& uncut_cells = multimesh.uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        volume += cell.volume();
      }

      // std::cout << "\t uncut volume "<< part_volume << ' ';

      // Cut cell volume given by quadrature rule
      const std::vector<unsigned int>& cut_cells = multimesh.cut_cells(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
        // std::cout << "Cut cell in part " << part << ": " << *it << std::endl;
        const quadrature_rule& qr = multimesh.quadrature_rule_cut_cell(part, *it);
        // std::cout << "QR: " << qr.first.size() << ", " << qr.second.size() << std::endl;
        for (std::size_t i = 0; i < qr.second.size(); ++i)
        {
          volume += qr.second[i];
        }
      }
    }

    return volume;
  }
  //------------------------------------------------------------------------------
  void get_uncut_cells_cgal(const MultiMesh& multimesh, std::vector<std::vector<std::size_t>>& uncut_cells) const
  {
    //typedef CGAL::Exact_predicates_exact_constructions_kernel ExactKernel;
    typedef CGAL::Exact_predicates_inexact_constructions_kernel ExactKernel;
    typedef CGAL::Triangle_2<ExactKernel> Triangle_2;
    typedef CGAL::Point_2<ExactKernel> Point_2;
    
    uncut_cells.reserve(multimesh.num_parts());

    for (std::size_t i = 0; i < multimesh.num_parts(); i++)
    {
      uncut_cells.push_back(std::vector<std::size_t>());
      std::vector<std::size_t>& current_uncut_cells = uncut_cells.back();

      std::shared_ptr<const Mesh> current_mesh = multimesh.part(i);
      const MeshGeometry& current_geometry = current_mesh->geometry();
      
      for (CellIterator cit(*current_mesh); !cit.end(); ++cit)
      {
        // Test every cell against every cell in overlaying meshes
        Triangle_2 current_cell(Point_2(current_geometry.x(cit->entities(0)[0], 0),
                                        current_geometry.x(cit->entities(0)[0], 1)),
                                Point_2(current_geometry.x(cit->entities(0)[1], 0),
                                        current_geometry.x(cit->entities(0)[1], 1)),
                                Point_2(current_geometry.x(cit->entities(0)[2], 0),
                                        current_geometry.x(cit->entities(0)[2], 1)));
        bool is_uncut = true;
        for (std::size_t j = i+1; j < multimesh.num_parts(); j++)
        {
          std::shared_ptr<const Mesh> other_mesh = multimesh.part(j);
          const MeshGeometry& other_geometry = other_mesh->geometry();
          for (CellIterator cit_other(*other_mesh); !cit_other.end(); ++cit_other)
          {
            Triangle_2 other_cell(Point_2(other_geometry.x(cit->entities(0)[0], 0),
                                          other_geometry.x(cit->entities(0)[0], 1)),
                                  Point_2(other_geometry.x(cit->entities(0)[1], 0),
                                          other_geometry.x(cit->entities(0)[1], 1)),
                                  Point_2(other_geometry.x(cit->entities(0)[2], 0),
                                          other_geometry.x(cit->entities(0)[2], 1)));
            if (do_intersect(current_cell, other_cell))
            {
              is_uncut = false;
              break;
            }
          }

          if (!is_uncut)
            break;
        }

        if (is_uncut)
        {
          current_uncut_cells.push_back(cit->index());
        }
      }
    }
  }
  //------------------------------------------------------------------------------
  void test_multiple_meshes_with_rotation()
  {
    set_log_level(DBG);

    dolfin::seed(0);

    const double h = 0.5;
    UnitSquareMesh background_mesh((int)std::round(1./h),
				   (int)std::round(1./h));

    MultiMesh multimesh;
    multimesh.add(background_mesh);

    const std::size_t Nmeshes = 8;

    std::size_t i = 0;
    while (i < Nmeshes)
    {
      double x0 = dolfin::rand();
      double x1 = dolfin::rand();
      if (x0 > x1) std::swap(x0, x1);
      double y0 = dolfin::rand();
      double y1 = dolfin::rand();
      if (y0 > y1) std::swap(y0, y1);
      const double v = dolfin::rand()*90; // initial rotation
      const double speed = dolfin::rand()-0.5; // initial speed
       
      const double cx = (x0+x1) / 2;
      const double cy = (y0+y1) / 2;
      double xr, yr;
      rotate(x0, y0, cx, cy, v, xr, yr);
      if (xr > 0 and xr < 1 and yr > 0 and yr < 1)
      {
	rotate(x0, y1, cx, cy, v, xr, yr);
	if (xr > 0 and xr < 1 and yr > 0 and yr < 1)
	{
	  rotate(x1, y0, cx, cy, v, xr, yr);
	  if (xr > 0 and xr < 1 and yr > 0 and yr < 1)
	  {
	    rotate(x1, y1, cx, cy, v, xr, yr);
	    if (xr > 0 and xr < 1 and yr > 0 and yr < 1)
	    {
              std::shared_ptr<Mesh> mesh(new RectangleMesh(x0, y0, x1, y1,
                                                           std::max((int)std::round((x1-x0)/h), 1),
                                                           std::max((int)std::round((y1-y0)/h), 1)));
              mesh->rotate(v);              
              multimesh.add(mesh);
	      i++;
	    }
	  }
	}
      }
    }

    multimesh.build();

    std::vector<std::vector<std::size_t>> uncut_cells_cgal;
    get_uncut_cells_cgal(multimesh, uncut_cells_cgal);
    for (std::size_t i = 0; i < uncut_cells_cgal.size(); i++)
    {
      const std::vector<std::size_t>& current_uncut = uncut_cells_cgal[i];
      std::cout << "Uncut cells in part " << i << ": ";
      for (auto it = current_uncut.begin(); it != current_uncut.end(); it++)
        std::cout << *it << " ";
      std::cout << std::endl;
    }
    
    if (MULTIMESH_DEBUG_OUTPUT)
    {
      std::cout << multimesh.plot_matplotlib() << std::endl;
    }

    // Exact volume is known
    const double exact_volume = 1;
    const double volume = compute_volume(multimesh);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }


};

int main()
{
  // Test not workin in parallel
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Skipping unit test in parallel.");
    info("OK");
    return 0;
  }

  CPPUNIT_TEST_SUITE_REGISTRATION(MultiMeshesCgal);
  DOLFIN_TEST;
}
