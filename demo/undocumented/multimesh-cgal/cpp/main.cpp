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


#define MULTIMESH_DEBUG_OUTPUT 0

using namespace dolfin;

enum CELL_STATUS
{
  UNKNOWN,
  COVERED,
  CUT,
  UNCUT
};

std::string cell_status_str(CELL_STATUS cs)
{
  switch(cs)
  {
   case UNKNOWN :
    return "UNKNOWN";
   case COVERED :
    return "COVERED";
   case CUT :
    return "CUT    ";
   case UNCUT :
    return "UNCUT  ";
  }
}

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
// Compute volume contributions from each cell
void compute_volume(const MultiMesh& multimesh,
                    std::vector<std::vector<std::pair<CELL_STATUS, double> > >& cells_status)
{
  cells_status.reserve(multimesh.num_parts());    

  // Compute contribution from all parts
  for (std::size_t part = 0; part < multimesh.num_parts(); part++)
  {
    // std::cout << "Testing part " << part << std::endl;
    cells_status.push_back(std::vector<std::pair<CELL_STATUS, double> >());
    std::vector<std::pair<CELL_STATUS, double> >& current_cells_status = cells_status.back();

    std::shared_ptr<const Mesh> current_mesh = multimesh.part(part);
    current_cells_status.resize(current_mesh->num_cells());
    // std::cout << "Number of cells: " << current_cells_status.size() << std::endl;

    // Uncut cell volume given by function volume
    {
      const std::vector<unsigned int>& uncut_cells = multimesh.uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        current_cells_status[*it] = std::make_pair(UNCUT, cell.volume());
      }
    }
      
    // Cut cell volume given by quadrature rule
    {
      const std::vector<unsigned int>& cut_cells = multimesh.cut_cells(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
        // std::cout << "Cut cell in part " << part << ": " << *it << std::endl;
        double volume = 0;
        const quadrature_rule& qr = multimesh.quadrature_rule_cut_cell(part, *it);
        // std::cout << "QR: " << qr.first.size() << ", " << qr.second.size() << std::endl;
        for (std::size_t i = 0; i < qr.second.size(); ++i)
        {
          volume += qr.second[i];
        }
        current_cells_status[*it] = std::make_pair(CUT, volume);
      }
    }

    {
      const std::vector<unsigned int>& covered_cells = multimesh.covered_cells(part);
      for (auto it = covered_cells.begin(); it != covered_cells.end(); ++it)
      {
        current_cells_status[*it] = std::make_pair(COVERED, 0.);
      }
    }
  }
}
//------------------------------------------------------------------------------
void get_cells_status_cgal(const MultiMesh& multimesh,
                           std::vector<std::vector<std::pair<CELL_STATUS, FT> > >& cells_status)
{
  cells_status.reserve(multimesh.num_parts());

  FT volume = 0;

  for (std::size_t i = 0; i < multimesh.num_parts(); i++)
  {
    // std::cout << "Testing part " << i << std::endl;
    cells_status.push_back(std::vector<std::pair<CELL_STATUS, FT> >());
    std::vector<std::pair<CELL_STATUS, FT> >& current_cells_status = cells_status.back();

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
      if (current_cell.orientation() == CGAL::CLOCKWISE)
      {
        //std::cout << "Orig: " << current_cell << std::endl;
        current_cell = current_cell.opposite();
        //std::cout << "Opposite: " << current_cell << std::endl;
      }
      Polygon_set_2 polygon_set;
      {
        std::vector<Point_2> vertices;
        vertices.push_back(current_cell[0]);
        vertices.push_back(current_cell[1]);
        vertices.push_back(current_cell[2]);

        Polygon_2 p(vertices.begin(), vertices.end());
        polygon_set.insert(p);
      }

      // std::cout << "  Testing cell: " << current_cell << std::endl;
      bool is_uncut = true;
      for (std::size_t j = i+1; j < multimesh.num_parts(); j++)
      {
        // std::cout << "    Testing against part " << j << std::endl;
        std::shared_ptr<const Mesh> other_mesh = multimesh.part(j);
        const MeshGeometry& other_geometry = other_mesh->geometry();
        for (CellIterator cit_other(*other_mesh); !cit_other.end(); ++cit_other)
        {
          std::vector<Point_2> vertices;
          Point_2 p0(other_geometry.x(cit_other->entities(0)[0], 0),
                     other_geometry.x(cit_other->entities(0)[0], 1));
          Point_2 p1(other_geometry.x(cit_other->entities(0)[1], 0),
                     other_geometry.x(cit_other->entities(0)[1], 1));
          Point_2 p2(other_geometry.x(cit_other->entities(0)[2], 0),
                     other_geometry.x(cit_other->entities(0)[2], 1));

          vertices.push_back(p0);
          if (Line_2(p0, p1).has_on_positive_side(p2))
          {
            vertices.push_back(p1);
            vertices.push_back(p2);
          }
          else
          {
            vertices.push_back(p2);
            vertices.push_back(p1);
          }
          Polygon_2 p(vertices.begin(), vertices.end());
          polygon_set.difference(p);
        }
      }

      std::vector<Polygon_with_holes_2> result;
      polygon_set.polygons_with_holes(std::back_inserter(result));

      if (result.size() == 0)
      {
        current_cells_status.push_back(std::make_pair(COVERED, 0.0));
        //std::cout << "    Covered" << std::endl;
      }
      else
      {
        // if (result.size() > 1)
        //   std::cout << "!!!!!!!! Several polygons !!!!!!!" << std::endl;

        Polygon_2::Vertex_const_iterator v = result[0].outer_boundary().vertices_begin();
        Polygon_2::Vertex_const_iterator v_end = result[0].outer_boundary().vertices_end();
        const std::size_t num_vertices = std::distance(v, v_end);
        const Point_2& v0 = *v; ++v;
        const Point_2& v1 = *v; ++v;
        const Point_2& v2 = *v;

        if (result.size() == 1 &&
            result[0].holes_begin() == result[0].holes_end() &&
            num_vertices == 3 &&
            Triangle_2(v0, v1, v2) == current_cell)
        {
          current_cells_status.push_back(std::make_pair(UNCUT,
                                                        result[0].outer_boundary().area()));
          // std::cout << "    Uncut" << std::endl;
        }
        else
        {
          FT current_volume = 0;

          for(auto pit = result.begin(); pit != result.end(); pit++)
          {
            const Polygon_2& outerboundary = pit->outer_boundary();
            current_volume += outerboundary.area();
            // std::cout << "    Polygon ";
            // for (auto it = outerboundary.vertices_begin(); it != outerboundary.vertices_end(); it++) std::cout << *it << ", ";
            // std::cout << std::endl;

            for (auto it = pit->holes_begin(); it != pit->holes_end(); it++)
              current_volume += it->area();
          }
          current_cells_status.push_back(std::make_pair(CUT, current_volume));
          // std::cout << "    Cut" << std::endl;
        }
      }
    }
  }
}
//------------------------------------------------------------------------------
void test_multiple_meshes_with_rotation()
{
  set_log_level(DBG);

  const double h = 0.5;

  MultiMesh multimesh;
  {
    std::shared_ptr<Mesh> background_mesh(new UnitSquareMesh((int)std::round(1./h),
                                                             (int)std::round(1./h)));
    multimesh.add(background_mesh);
  }

  {
    const double x0 = 0.40022862209017789903;
    const double y0 = 0.28331474600514150453;
    const double x1 = 0.89152945200518218805;
    const double y1 = 0.35245834726489072564;
    std::shared_ptr<Mesh> mesh(new RectangleMesh(Point(x0, y0), Point(x1, y1),
                                                 std::max((int)std::round((x1-x0)/h), 1),
                                                 std::max((int)std::round((y1-y0)/h), 1)));
    mesh->rotate(72.695206800799439861);
    multimesh.add(mesh);
  }

  {
    const double x0 = 0.37520697637237931943;
    const double y0 = 0.51253536414007438982;
    const double x1 = 0.76024873636674539235;
    const double y1 = 0.66772376078540629507;
    std::shared_ptr<Mesh> mesh(new RectangleMesh(Point(x0, y0), Point(x1, y1),
                                                 std::max((int)std::round((x1-x0)/h), 1),
                                                 std::max((int)std::round((y1-y0)/h), 1)));
    mesh->rotate(47.844579074459417711);
    multimesh.add(mesh);
  }

  {
    const double x0 = 0.35404867974764142602;
    const double y0 = 0.16597416632155614913;
    const double x1 = 0.63997881656511634851;
    const double y1 = 0.68786139026650294781;
    std::shared_ptr<Mesh> mesh(new RectangleMesh(Point(x0, y0), Point(x1, y1),
                                                 std::max((int)std::round((x1-x0)/h), 1),
                                                 std::max((int)std::round((y1-y0)/h), 1)));
    mesh->rotate(39.609407484349517858);
    multimesh.add(mesh);
  }

  {
    const double x0 = 0.33033712968711609337;
    const double y0 = 0.22896817104377231722;
    const double x1 = 0.82920109332967595339;
    const double y1 = 0.89337241458397931293;
    std::shared_ptr<Mesh> mesh(new RectangleMesh(Point(x0, y0), Point(x1, y1),
                                                 std::max((int)std::round((x1-x0)/h), 1),
                                                 std::max((int)std::round((y1-y0)/h), 1)));
    mesh->rotate(31.532416069662392744);
    multimesh.add(mesh);
  }

  {
    const double x0 = 0.58864013319306085492;
    const double y0 = 0.65730403953106331105;
    const double x1 = 0.95646825291051917883;
    const double y1 = 0.85867632592966613991;
    std::shared_ptr<Mesh> mesh(new RectangleMesh(Point(x0, y0), Point(x1, y1),
                                                 std::max((int)std::round((x1-x0)/h), 1),
                                                 std::max((int)std::round((y1-y0)/h), 1)));
    mesh->rotate(39.560392754879032395);
    multimesh.add(mesh);
  }

  {
    const double x0 = 0.28105941241656401397;
    const double y0 = 0.30745787374091237965;
    const double x1 = 0.61959648394007071914;
    const double y1 = 0.78600209801737319637;
    std::shared_ptr<Mesh> mesh(new RectangleMesh(Point(x0, y0), Point(x1, y1),
                                                 std::max((int)std::round((x1-x0)/h), 1),
                                                 std::max((int)std::round((y1-y0)/h), 1)));
    mesh->rotate(40.233022128340330426);
    multimesh.add(mesh);
  }

  multimesh.build();
  
  std::cout << multimesh.plot_matplotlib() << std::endl;
  std::cout << "Done building multimesh" << std::endl;
  /* ---------------- Done creating multimesh ----------------------- */

  // Compute volume of each cell using cgal
  std::vector<std::vector<std::pair<CELL_STATUS, FT>>> cell_status_cgal;
  get_cells_status_cgal(multimesh, cell_status_cgal);
  std::cout << "Done computing volumes with cgal" << std::endl;

  // Compute volume of each cell using dolfin::MultiMesh
  std::vector<std::vector<std::pair<CELL_STATUS, double> > > cell_status_multimesh;
  compute_volume(multimesh, cell_status_multimesh);
  std::cout << "Done computing volumes with multimesh" << std::endl;
  
  FT cgal_volume = 0.;
  double multimesh_volume = 0.;

  dolfin_assert(cell_status_cgal.size() == cell_status_multimesh.size());
  for (std::size_t i = 0; i < cell_status_cgal.size(); i++)
  {
    const std::vector<std::pair<CELL_STATUS, FT> >& current_cgal = cell_status_cgal[i];
    const std::vector<std::pair<CELL_STATUS, double> >& current_multimesh = cell_status_multimesh[i];

    dolfin_assert(current_cgal.size() == current_multimesh.size());
      
    std::cout << "Cells in part " << i << ": " << std::endl;
    for (std::size_t j = 0; j < current_cgal.size(); j++)
    {
      std::cout << "  Cell " << j << std::endl;
      std::cout << "    Multimesh: " << cell_status_str(current_multimesh[j].first) << " (" << current_multimesh[j].second << ")" << std::endl;
      std::cout << "    CGAL:      " << cell_status_str(current_cgal[j].first) << " (" << current_cgal[j].second << ")" << std::endl;
      std::cout << "      Diff:    " << (current_cgal[j].second - current_multimesh[j].second) << std::endl;
      cgal_volume += current_cgal[j].second;
      multimesh_volume += current_multimesh[j].second;
      // dolfin_assert(near(current_cgal[j].second, current_multimesh[j].second, DOLFIN_EPS_LARGE));
      // dolfin_assert(current_cgal[j].first == current_multimesh[j].first);
    }
    std::cout << std::endl;
  }

  // Exact volume is known
  const FT exact_volume = 1;
    
  std::cout << "Total volume" << std::endl;
  std::cout << "------------" << std::endl;
  std::cout << "Multimesh: " << multimesh_volume << ", error: " << (exact_volume-multimesh_volume) << std::endl;
  std::cout << "CGAL:      " << cgal_volume << ", error: " << (exact_volume-cgal_volume) << std::endl;

  //dolfin_assert(near(exact_volume, volume, DOLFIN_EPS_LARGE));
}

int main(int argc, char** argv)
{
  test_multiple_meshes_with_rotation();
}
