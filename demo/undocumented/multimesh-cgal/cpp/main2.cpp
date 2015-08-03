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

#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/math/basic.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/generation/UnitSquareMesh.h>

#include <CGAL/Triangle_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/intersection_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Polygon_set_2.h>

typedef CGAL::Exact_predicates_exact_constructions_kernel ExactKernel;
//typedef CGAL::Exact_predicates_inexact_constructions_kernel ExactKernel;
typedef CGAL::Point_2<ExactKernel>                Point_2;
typedef CGAL::Vector_2<ExactKernel>               Vector_2;
typedef CGAL::Segment_2<ExactKernel>              Segment_2;
typedef CGAL::Triangle_2<ExactKernel>             Triangle_2;
typedef CGAL::Line_2<ExactKernel>                 Line_2;
typedef CGAL::Polygon_2<ExactKernel>              Polygon_2;
typedef Polygon_2::Vertex_const_iterator          Vertex_const_iterator;
typedef CGAL::Polygon_with_holes_2<ExactKernel>   Polygon_with_holes_2;
typedef Polygon_with_holes_2::Hole_const_iterator Hole_const_iterator;
typedef CGAL::Polygon_set_2<ExactKernel>          Polygon_set_2;
typedef std::pair<Point_2, ExactKernel::FT> cgal_QR;

// typedef std::vector<Point> Simplex;
// typedef std::vector<Simplex> Polyhedron;


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
        std::cout << "Cut cell in part " << part << ": " << *it << std::endl;
        double volume = 0;
        const quadrature_rule& qr = multimesh.quadrature_rule_cut_cell(part, *it);
        std::cout << "  QR: " << qr.first.size() << ", " << qr.second.size() << std::endl;
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
void compute_quadrature_rules_overlap_cgal(const MultiMesh& multimesh,
                                           std::vector<std::map<unsigned int, cgal_QR>>& qr_rules_overlap)
{

  std::cout << "Computing overlap quadrature rules" << std::endl;
  
  qr_rules_overlap.clear();

  std::ofstream debug_file("cgal-output.txt");
  debug_file << std::setprecision(20);

  // Iterate over all parts
  for (std::size_t cut_part = 0; cut_part < multimesh.num_parts(); cut_part++)
  {
    std::cout << "----- cut part: " << cut_part << std::endl;
    std::shared_ptr<const Mesh> cut_mesh = multimesh.part(cut_part);
    const MeshGeometry& cut_mesh_geometry = cut_mesh->geometry();
    

    // Iterate over cut cells for current part
    for (CellIterator cut_it(*cut_mesh); !cut_it.end(); ++cut_it)
    {
      // Test every cell against every cell in overlaying meshes
      Triangle_2 cut_cell(Point_2(cut_mesh_geometry.x(cut_it->entities(0)[0], 0),
                                  cut_mesh_geometry.x(cut_it->entities(0)[0], 1)),
                          Point_2(cut_mesh_geometry.x(cut_it->entities(0)[1], 0),
                                  cut_mesh_geometry.x(cut_it->entities(0)[1], 1)),
                          Point_2(cut_mesh_geometry.x(cut_it->entities(0)[2], 0),
                                  cut_mesh_geometry.x(cut_it->entities(0)[2], 1)));
      if (cut_cell.orientation() == CGAL::CLOCKWISE)
        cut_cell = cut_cell.opposite();

      for (std::size_t cutting_part = cut_part+1; cutting_part < multimesh.num_parts(); cutting_part++)
      {
        std::shared_ptr<const Mesh> cutting_mesh = multimesh.part(cutting_part);
        const MeshGeometry& cutting_mesh_geometry = cutting_mesh->geometry();

        for (CellIterator cutting_it(*cutting_mesh); !cutting_it.end(); ++cutting_it)
        {
          // Test every cell against every cell in overlaying meshes
          Triangle_2 cutting_cell(Point_2(cutting_mesh_geometry.x(cutting_it->entities(0)[0], 0),
                                          cutting_mesh_geometry.x(cutting_it->entities(0)[0], 1)),
                                  Point_2(cutting_mesh_geometry.x(cutting_it->entities(0)[1], 0),
                                          cutting_mesh_geometry.x(cutting_it->entities(0)[1], 1)),
                                  Point_2(cutting_mesh_geometry.x(cutting_it->entities(0)[2], 0),
                                          cutting_mesh_geometry.x(cutting_it->entities(0)[2], 1)));
          if (cutting_cell.orientation() == CGAL::CLOCKWISE)
            cutting_cell = cutting_cell.opposite();

          if (CGAL::do_intersect(cut_cell, cutting_cell))
          {
            std::cout << "Intersects: (" << cut_part << ", " << cut_it->index() << ") and (" << cutting_part << ", " << cutting_it->index() << ")" << std::endl;

            // Data structure for the overlap quadrature rule
            std::vector<cgal_QR> overlap_qr;

            auto cell_intersection = CGAL::intersection(cut_cell, cutting_cell);
            dolfin_assert(cell_intersection);
              // alternatively:
            if (const Segment_2* s = boost::get<Segment_2>(&*cell_intersection)) {
              // handle segment
              std::cout << "  Segment" << std::endl;
            }
            else if (const Point_2* p = boost::get<Point_2>(&*cell_intersection))
            {
              // handle point
              std::cout << "  Point" << std::endl;
            }
            else if (const Triangle_2* t = boost::get<Triangle_2>(&*cell_intersection))
            {
              // handle triangle intersection
              // Print the triangles in a reproducible order
              std::vector<std::pair<std::pair<ExactKernel::FT, ExactKernel::FT>, std::size_t>> v{ std::make_pair(std::make_pair(t->vertex(0)[0], t->vertex(0)[1]), 0),
                                                                                                  std::make_pair(std::make_pair(t->vertex(1)[0], t->vertex(1)[1]), 1),
                                                                                                  std::make_pair(std::make_pair(t->vertex(2)[0], t->vertex(2)[1]), 2)};
              std::sort(v.begin(), v.end());
              debug_file <<  "(" << cut_part << "," << cut_it->index() << ") (" << cutting_part << "," << cutting_it->index() << ") : "
                         << t->vertex(v[0].second) << ", "
                         << t->vertex(v[1].second) << ", "
                         << t->vertex(v[2].second) << std::endl;
            }
            else 
            {
              const std::vector<Point_2>* polygon = boost::get<std::vector<Point_2>>(&*cell_intersection);
              dolfin_assert(polygon);

              // Now triangulate polygon the same way as multimesh does it
              // geometry/IntersectionTriangulation.cpp:598

              // Find left-most point (smallest x-coordinate)
              // Use y-coordinate if x-coordinates are exactly equal.
              // TODO: Does this work in 3D? Then also include z-coordinate in the
              // comparison.
              std::size_t i_min = 0;
              Point_2 point_min = (*polygon)[0];
              for (std::size_t i = 1; i < polygon->size(); i++)
              {
                //const double x = points[i].x();
                if (point_min.x() < (*polygon)[i].x() || (point_min.x() == (*polygon)[i].x() && point_min.y() < (*polygon)[i].y()))
                {
                  point_min = (*polygon)[i];
                  i_min = i;
                }
              }

              // Compute signed squared cos of angle with (0, 1) from i_min to all points
              std::vector<std::pair<ExactKernel::FT, std::size_t>> order;
              for (std::size_t i = 0; i < polygon->size(); i++)
              {
                // Skip left-most point used as origin
                if (i == i_min)
                  continue;

                // Compute vector to point
                const Vector_2 v = (*polygon)[i] - (*polygon)[i_min];

                // Compute square cos of angle
                const ExactKernel::FT cos2 = (v.y() < 0.0 ? -1.0 : 1.0)*v.y()*v.y() / v.squared_length();

                // Store for sorting
                order.push_back(std::make_pair(cos2, i));
              }

              // Sort points based on angle
              std::sort(order.begin(), order.end());

              std::cout << "Order: ";
              for(const std::pair<ExactKernel::FT, std::size_t>& item : order)
              {
                std::cout << item.second << " ";
              }
              std::cout << std::endl;

              // Triangulate polygon by connecting i_min with the ordered points
              //triangulation.reserve((points.size() - 2)*3*2);
              debug_file <<  "(" << cut_part << "," << cut_it->index() << ") (" << cutting_part << "," << cutting_it->index() << ") : ";
              const Point_2& p0 = (*polygon)[i_min];
              for (std::size_t i = 0; i < polygon->size() - 2; i++)
              {
                const Point_2& p1 = (*polygon)[order[i].second];
                const Point_2& p2 = (*polygon)[order[i + 1].second];
                debug_file << p0 << ", " << p1 << ", " << p2 << ", ";
              }
              debug_file << std::endl;
            }
          }
        }
      }
    }
  }
  debug_file.close();
}
//------------------------------------------------------------------------------
void compute_volume_cgal(const MultiMesh& multimesh,
                         std::vector<std::vector<std::pair<CELL_STATUS, double> > >& cells_status)
{
  std::vector<std::map<unsigned int, cgal_QR>> qr_rules_overlap;
  compute_quadrature_rules_overlap_cgal(multimesh, qr_rules_overlap);
  
  cells_status.reserve(multimesh.num_parts());

  ExactKernel::FT volume = 0;

  for (std::size_t i = 0; i < multimesh.num_parts(); i++)
  {
    // std::cout << "Testing part " << i << std::endl;
    cells_status.push_back(std::vector<std::pair<CELL_STATUS, double> >());
    std::vector<std::pair<CELL_STATUS, double> >& current_cells_status = cells_status.back();

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
                                                        CGAL::to_double(result[0].outer_boundary().area())));
          // std::cout << "    Uncut" << std::endl;
        }
        else
        {
          ExactKernel::FT current_volume = 0;

          for(auto pit = result.begin(); pit != result.end(); pit++)
          {
            const Polygon_2& outerboundary = pit->outer_boundary();
            current_volume += outerboundary.area();
            // std::cout << "    Polygon ";
            // for (auto it = outerboundary.vertices_begin(); it != outerboundary.vertices_end(); it++) std::cout << *it << ", ";
            // std::cout << std::endl;

            for (auto it = pit->holes_begin(); it != pit->holes_end(); it++)
              current_volume -= it->area();
          }
          current_cells_status.push_back(std::make_pair(CUT, CGAL::to_double(current_volume)));
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

  dolfin::seed(0);

  const double h = 0.5;
  UnitSquareMesh background_mesh((int)std::round(1./h),
                                 (int)std::round(1./h));

  MultiMesh multimesh;
  multimesh.add(background_mesh);

  const std::size_t Nmeshes = 8;

  /* ---------------- Create multimesh ------------------------- */
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

            // The error happends in mesh from mesh 2 and up
            if (i > 1)
              multimesh.add(mesh);
            i++;
          }
        }
      }
    }
  }

  multimesh.build();
  
  // std::cout << multimesh.plot_matplotlib() << std::endl;
  std::cout << "Done building multimesh" << std::endl;
  /* ---------------- Done creating multimesh ----------------------- */

  // Compute volume of each cell using cgal
  std::vector<std::vector<std::pair<CELL_STATUS, double>>> cell_status_cgal;
  compute_volume_cgal(multimesh, cell_status_cgal);
  std::cout << "Done computing volumes with cgal" << std::endl;

  // Compute volume of each cell using dolfin::MultiMesh
  std::vector<std::vector<std::pair<CELL_STATUS, double>>> cell_status_multimesh;
  compute_volume(multimesh, cell_status_multimesh);
  std::cout << "Done computing volumes with multimesh" << std::endl;
  
  double cgal_volume = 0.;
  double multimesh_volume = 0.;

  dolfin_assert(cell_status_cgal.size() == cell_status_multimesh.size());
  for (std::size_t i = 0; i < cell_status_cgal.size(); i++)
  {
    const std::vector<std::pair<CELL_STATUS, double> >& current_cgal = cell_status_cgal[i];
    const std::vector<std::pair<CELL_STATUS, double> >& current_multimesh = cell_status_multimesh[i];

    dolfin_assert(current_cgal.size() == current_multimesh.size());
      
    std::cout << "Cells in part " << i << ": " << std::endl;
    for (std::size_t j = 0; j < current_cgal.size(); j++)
    {
      std::cout << "  Cell " << j << std::endl;
      std::cout << "    Multimesh: " << cell_status_str(current_multimesh[j].first) << " (" << current_multimesh[j].second << ")" << std::endl;
      std::cout << "    CGAL:      " << cell_status_str(current_cgal[j].first) << " (" << current_cgal[j].second << ")" << std::endl;
      std::cout << "      Diff:    " << std::abs(current_cgal[j].second - current_multimesh[j].second) << std::endl;
      cgal_volume += current_cgal[j].second;
      multimesh_volume += current_multimesh[j].second;
    }
    std::cout << std::endl;
  }

  // Exact volume is known
  const double exact_volume = 1;
    
  std::cout << "Total volume" << std::endl;
  std::cout << "------------" << std::endl;
  std::cout << "Multimesh: " << multimesh_volume << ", error: " << std::abs(exact_volume-multimesh_volume) << std::endl;
  std::cout << "CGAL:      " << cgal_volume << ", error: " << std::abs(exact_volume-cgal_volume) << std::endl;
}

int main(int argc, char** argv)
{
  test_multiple_meshes_with_rotation();
}
