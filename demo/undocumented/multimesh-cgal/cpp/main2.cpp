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
#include <dolfin/math/basic.h>
#include <dolfin/mesh/Cell.h>

#include <CGAL/Triangle_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/intersection_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Polygon_set_2.h>

#include "common.h"

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
typedef std::vector<std::pair<Point_2, ExactKernel::FT>> cgal_QR;

typedef std::vector<Triangle_2> Polygon;


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

static bool debug_output = false;

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
std::pair<Point_2, ExactKernel::FT> cgal_compute_quadrature_rule(Triangle_2 t, ExactKernel::FT factor)
{
  const Vector_2 a = t[1]-t[0];
  const Vector_2 b = t[2]-t[0];

  // Compute double the area of the triangle
  const ExactKernel::FT det = CGAL::abs(a.x()*b.y() - a.y()*b.x());

  // qr.push_back(std::make_pair( CGAL::ORIGIN + (t[0]-CGAL::ORIGIN)/3 + (t[1]-CGAL::ORIGIN)/3 + (t[2]-CGAL::ORIGIN)/3,
  return std::make_pair( CGAL::centroid(t), factor*det/2 );
}
//------------------------------------------------------------------------------
Polygon compute_intersection_triangulation(Triangle_2 t1, Triangle_2 t2)
{
  std::cout << std::setprecision(25);
  if (debug_output)
  {
    std::cout << "InterscetionTriangulation" << std::endl;
    std::cout << t1[0] << " " << t1[1] << " " << t1[2] << std::endl;
    std::cout << t2[0] << " " << t2[1] << " " << t2[2] << std::endl;
  }
  
  Polygon intersection;
  if (CGAL::do_intersect(t1, t2))
  {
    auto cell_intersection = CGAL::intersection(t1, t2);
    dolfin_assert(cell_intersection);

    if (const Segment_2* s = boost::get<Segment_2>(&*cell_intersection))
    {
      // segment intersection
      // do nothing
    }
    else if (const Point_2* p = boost::get<Point_2>(&*cell_intersection))
    {
      // point intersection
      // do nothing
    }
    else if (const Triangle_2* t = boost::get<Triangle_2>(&*cell_intersection))
    {
      // handle triangle intersection
      // Print the triangles in a reproducible order
      std::vector<std::pair<std::pair<ExactKernel::FT, ExactKernel::FT>, std::size_t>> v{ std::make_pair(std::make_pair(t->vertex(0)[0], t->vertex(0)[1]), 0),
                                                                                          std::make_pair(std::make_pair(t->vertex(1)[0], t->vertex(1)[1]), 1),
                                                                                          std::make_pair(std::make_pair(t->vertex(2)[0], t->vertex(2)[1]), 2)};
      std::sort(v.begin(), v.end());
      // debug_file <<  "(" << cut_part << "," << cut_it->index() << ") (" << cutting_part << "," << cutting_it->index() << ") : "
      //            << t->vertex(v[0].second) << ", "
      //            << t->vertex(v[1].second) << ", "
      //            << t->vertex(v[2].second) << std::endl;
      intersection.push_back(Triangle_2(t->vertex(v[0].second),
                                        t->vertex(v[1].second),
                                        t->vertex(v[2].second)));
    }
    else 
    {
      const std::vector<Point_2>* polygon = boost::get<std::vector<Point_2>>(&*cell_intersection);
      dolfin_assert(polygon);
      if (debug_output) std::cout << "Polygon size: " << polygon->size() << std::endl;
      
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

      // std::cout << "Order: ";
      // for(const std::pair<ExactKernel::FT, std::size_t>& item : order)
      // {
      //   std::cout << item.second << " ";
      // }
      // std::cout << std::endl;

      // Triangulate polygon by connecting i_min with the ordered points

      // debug_file <<  "(" << cut_part << "," << cut_it->index() << ") (" << cutting_part << "," << cutting_it->index() << ") : ";
      const Point_2& p0 = (*polygon)[i_min];
      for (std::size_t i = 0; i < polygon->size() - 2; i++)
      {
        const Point_2& p1 = (*polygon)[order[i].second];
        const Point_2& p2 = (*polygon)[order[i + 1].second];
        // debug_file << p0 << ", " << p1 << ", " << p2 << ", ";
        intersection.push_back(Triangle_2(p0, p1, p2));
      }
      // debug_file << std::endl;
    }
  }
  return intersection;
}
//------------------------------------------------------------------------------
std::vector<cgal_QR> compute_cell_quadrature_rule(std::size_t cut_part,
                                                  unsigned int cut_cell_index)
{
                                                  
}
//------------------------------------------------------------------------------
void compute_quadrature_rules_overlap_cgal(const MultiMesh& multimesh,
                                           std::vector<std::map<unsigned int, std::vector<cgal_QR>>>& qr_rules_overlap)
{
  std::cout << "CGAL: Computing overlap quadrature rules" << std::endl;
  
  qr_rules_overlap.clear();

  // std::ofstream debug_file("cgal-output.txt");
  // debug_file << std::setprecision(20);

  // Store the intersection from the lower layers as sum of (disjoint) triangles
  // (which together form a polygon).
  // The key (first part of pair) are the layer numbers, eg. [0, 1, 3] is the
  // intersection of parts 0, 1 and 3. 

  // std::pair<std::size_t, unsigned int> debug_cell = std::make_pair(std::numeric_limits<std::size_t>::max(),
  //                                                                  std::numeric_limits<unsigned int>::max());
  std::pair<std::size_t, unsigned int> debug_cell = std::make_pair(1, 1);
  
  // Iterate over all parts
  for (std::size_t cut_part = 0; cut_part < multimesh.num_parts(); cut_part++)
  {
    qr_rules_overlap.push_back(std::map<unsigned int, std::vector<cgal_QR>>());
    std::map<unsigned int, std::vector<cgal_QR>>& qr_overlap_current_part = qr_rules_overlap.back();

    std::cout << "----- cut part: " << cut_part << std::endl;
    std::shared_ptr<const Mesh> cut_mesh = multimesh.part(cut_part);
    const MeshGeometry& cut_mesh_geometry = cut_mesh->geometry();
    
    // Iterate over cut cells for current part
    for (CellIterator cut_it(*cut_mesh); !cut_it.end(); ++cut_it)
    {
      debug_output = cut_part == debug_cell.first && cut_it->index() == debug_cell.second;
      std::cout << std::setprecision(20);

      // Test every cell against every cell in overlaying meshes
      Triangle_2 cut_cell(Point_2(cut_mesh_geometry.x(cut_it->entities(0)[0], 0),
                                  cut_mesh_geometry.x(cut_it->entities(0)[0], 1)),
                          Point_2(cut_mesh_geometry.x(cut_it->entities(0)[1], 0),
                                  cut_mesh_geometry.x(cut_it->entities(0)[1], 1)),
                          Point_2(cut_mesh_geometry.x(cut_it->entities(0)[2], 0),
                                  cut_mesh_geometry.x(cut_it->entities(0)[2], 1)));
      if (cut_cell.orientation() == CGAL::CLOCKWISE)
        cut_cell = cut_cell.opposite();

      if (debug_output) std::cout << "------- cut cell " << cut_it->index() << " : " << cut_cell[0] << ", " << cut_cell[1] << ", " << cut_cell[2] << std::endl;

      // Store the initial polygon which are the elements used in the
      // inclusion-exclusion principle
      std::vector<Polygon> initial_polygons;
      
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

          Polygon intersection = compute_intersection_triangulation(cut_cell, cutting_cell);
          if (intersection.size() > 0)
          {
            initial_polygons.push_back(intersection);
          }
        }
      }

      // Done computing initial polygons
      if (debug_output) std::cout << "  Computed " << initial_polygons.size() << " initial polygons" << std::endl;

      if (initial_polygons.size() > 0)
      {
        qr_overlap_current_part[cut_it->index()] = std::vector<cgal_QR>();
        std::vector<cgal_QR>& overlap_qr_current_cut_cell = qr_overlap_current_part[cut_it->index()];

        // Add initial stage (0) of inc-exc principle

        for (std::size_t i = 0; i < initial_polygons.size(); i++)
        {
          const Polygon& p = initial_polygons[i];
          if (debug_output) std::cout << "    Adding quadrature rule of initial polygon: " << i << std::endl;

          // Add a new quadrature rule corresponding to the overlap in initial_polygons[i]
          cgal_QR qr_cutting_cell;
          for (const Triangle_2& t : p)
          {
            const std::pair<Point_2, ExactKernel::FT> qr_current_triangle = cgal_compute_quadrature_rule(t, 1);

            if (debug_output ) std::cout << "      (" << t[0] << ", " << t[1] << ", " << t[2] << ") => " << qr_current_triangle.first << ", " << qr_current_triangle.second << std::endl;
            qr_cutting_cell.push_back(qr_current_triangle);
          }

          // for (const std::pair<Point_2, ExactKernel::FT>& point_rule : qr_cutting_cell)
          //   std::cout << "qr rule:   (" << point_rule.first << ") : " << point_rule.second << ", ";
          // std::cout << std::endl;

          overlap_qr_current_cut_cell.push_back(qr_cutting_cell);
        }

        std::vector<std::pair<std::vector<std::size_t>,
                              Polygon> > previous_intersections;

        // Initialize intersections from stage 0 for the inc-exc loop
        for (std::size_t i = 0; i < initial_polygons.size(); i++)
        {
          previous_intersections.push_back(std::make_pair(std::vector<std::size_t>(1, i),
                                                          initial_polygons[i]));
        }

        // The stage loop
        for (std::size_t stage = 1; stage < initial_polygons.size(); stage++)
        {
          if (debug_output) std::cout << "----------------- stage " << stage << " (" << previous_intersections.size() << ")" << std::endl;

          std::vector<std::pair<std::vector<std::size_t>,
                                Polygon> > new_intersections;

          // Loop over all intersections from the previous stage
          for (const std::pair<const std::vector<std::size_t>,
                               Polygon>& previous_polygon : previous_intersections)
          {
            // Loop over all initial polyhedra.
            for (std::size_t init_p = 0; init_p < initial_polygons.size(); init_p++)
            {
              const Polygon& initial_polygon = initial_polygons[init_p];
              if (init_p < previous_polygon.first[0])
              {
                if (debug_output)
                {
                  std::cout << "  Previous polygon ";
                  for (const std::size_t& i : previous_polygon.first)
                    std::cout << i;

                  std::cout << " with " << init_p << std::endl;
                }



                // We want to save the intersection of the previous
                // polyhedron and the initial polyhedron in one single
                // polyhedron.
                Polygon new_polygon;
                std::vector<std::size_t> new_keys;

                // Loop over all simplices in the initial_polyhedron and
                // the previous_polyhedron and append the intersection of
                // these to the new_polyhedron
                bool any_intersections = false;

                for (const Triangle_2& previous_simplex: previous_polygon.second)
                {
                  for (const Triangle_2& initial_simplex: initial_polygon)
                  {
                    // Compute the intersection
                    Polygon ii = compute_intersection_triangulation(previous_simplex, initial_simplex);
                    for (const Triangle_2& t : ii)
                    {
                      any_intersections = true;
                      new_polygon.push_back(t);
                    }

                    if (ii.size() && debug_output)
                    {
                      std::cout << "Triangulate " << initial_simplex[0] << ", " << initial_simplex[1] << ", " << initial_simplex[2]
                                << " and " << previous_simplex[0] << ", " << previous_simplex[1] << ", " << previous_simplex[2] << " ==> " << std::endl;
                      for (const Triangle_2& s : ii)
                      {
                        std::cout << "  " << s[0] << ", " << s[1] << ", " << s[2] << std::endl;
                      }

                    }
                  }
                }

                if (any_intersections)
                {
                  new_keys.push_back(init_p);
                  new_keys.insert(new_keys.end(),
                                  previous_polygon.first.begin(),
                                  previous_polygon.first.end());

                  // Save data
                  new_intersections.push_back(std::make_pair(new_keys, new_polygon));
                }
              }
            }
          }

          if (debug_output)
            std::cout << "    Number of new intersections: " << new_intersections.size() << std::endl;
          
          // Add quadrature rule with correct sign
          const double sign = std::pow(-1, stage);
          // quadrature_rule overlap_part_qr;
          cgal_QR qr_current_stage;
          for (const std::pair<std::vector<std::size_t>, Polygon>& p : new_intersections)
          {
            if (debug_output) std::cout << "      intersection" << std::endl;
            for (const Triangle_2& t : p.second)
            {
              std::pair<Point_2, ExactKernel::FT> qr_current_triangle = cgal_compute_quadrature_rule(t, sign);
              if (debug_output)
                if (debug_output ) std::cout << "      (" << t[0] << ", " << t[1] << ", " << t[2] << ") => " << qr_current_triangle.first << ", " << qr_current_triangle.second << std::endl;
              qr_current_stage.push_back(qr_current_triangle);
            }
          }

          // Add quadrature rule for overlap part
          overlap_qr_current_cut_cell.push_back(qr_current_stage);
          previous_intersections = new_intersections;
        }
      } // end if current_cell is cut
    } // end cut cell iterator
  } // end part loop
  std::cout << "Done computing quadrature rules with CGAL" << std::endl;
}
//------------------------------------------------------------------------------
void compute_volume_cgal(const MultiMesh& multimesh,
                         std::vector<std::vector<std::pair<CELL_STATUS, ExactKernel::FT> > >& cells_status)
{
  std::cout << "Computing volume with CGAL" << std::endl;

  cells_status.reserve(multimesh.num_parts());    

  std::vector<std::map<unsigned int, std::vector<cgal_QR>>> qr_rules_overlap;
  compute_quadrature_rules_overlap_cgal(multimesh, qr_rules_overlap);
  
  for (std::size_t part = 0; part < multimesh.num_parts(); part++)
  {
    if (debug_output) std::cout << "  Part " << part << ", cut cells: " << qr_rules_overlap[part].size() << std::endl;
    std::shared_ptr<const Mesh> cut_mesh = multimesh.part(part);
    const MeshGeometry& cut_mesh_geometry = cut_mesh->geometry();
    const std::map<unsigned int, std::vector<cgal_QR>>& qr_overlap_current_part = qr_rules_overlap[part];

    cells_status.push_back(std::vector<std::pair<CELL_STATUS, ExactKernel::FT> >());
    std::vector<std::pair<CELL_STATUS, ExactKernel::FT> >& cells_status_current_part = cells_status.back();

    std::shared_ptr<const Mesh> current_mesh = multimesh.part(part);
    cells_status_current_part.resize(current_mesh->num_cells());

    // Iterate over cut cells for current part
    for (CellIterator cut_it(*cut_mesh); !cut_it.end(); ++cut_it)
    {
      if (debug_output) std::cout << "    cell: " << cut_it->index() << std::endl;

      // Test every cell against every cell in overlaying meshes
      Triangle_2 cut_cell(Point_2(cut_mesh_geometry.x(cut_it->entities(0)[0], 0),
                                  cut_mesh_geometry.x(cut_it->entities(0)[0], 1)),
                          Point_2(cut_mesh_geometry.x(cut_it->entities(0)[1], 0),
                                  cut_mesh_geometry.x(cut_it->entities(0)[1], 1)),
                          Point_2(cut_mesh_geometry.x(cut_it->entities(0)[2], 0),
                                  cut_mesh_geometry.x(cut_it->entities(0)[2], 1)));
      if (cut_cell.orientation() == CGAL::CLOCKWISE)
        cut_cell = cut_cell.opposite();

      std::pair<CELL_STATUS, ExactKernel::FT>& status_current_cell = cells_status_current_part[cut_it->index()];
      status_current_cell.first = UNCUT;
      status_current_cell.second = CGAL::abs(cut_cell.area());

      std::map<unsigned int, std::vector<cgal_QR>>::const_iterator cutting_cells = qr_overlap_current_part.find(cut_it->index());
      if (cutting_cells != qr_overlap_current_part.end())
      {
        // const std::vector<cgal_QR>& qrs = *cutting_cells;
        if (debug_output) std::cout << "      cell is cut" << std::endl;
        ExactKernel::FT volume = 0;
        for (const cgal_QR& qr : cutting_cells->second)
        {
          if (debug_output) std::cout << "      QR: " << qr.size() << std::endl;
          for (const std::pair<Point_2, ExactKernel::FT>& qr_point_and_weight : qr)
          {
            status_current_cell.second -= qr_point_and_weight.second;
          }
        }
        status_current_cell.first = CUT;
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
  build_failing_case(multimesh);
  
  // std::cout << multimesh.plot_matplotlib() << std::endl;
  std::cout << "Done building multimesh" << std::endl;
  /* ---------------- Done creating multimesh ----------------------- */

  // Compute volume of each cell using cgal
  std::vector<std::vector<std::pair<CELL_STATUS, ExactKernel::FT>>> cell_status_cgal;
  compute_volume_cgal(multimesh, cell_status_cgal);
  std::cout << "Done computing volumes with cgal" << std::endl;

  std::cout << cell_status_cgal.size() << std::endl;
  for (const std::vector<std::pair<CELL_STATUS, ExactKernel::FT>>& v : cell_status_cgal)
  {
    std::cout << "  part size  " << v.size() << std::endl;
    for (const std::pair<CELL_STATUS, ExactKernel::FT>& c : v)
      std::cout << "    cell, status" << c.first << " : volume " << c.second << std::endl;
  }
  
  // Compute volume of each cell using dolfin::MultiMesh
  std::vector<std::vector<std::pair<CELL_STATUS, double>>> cell_status_multimesh;
  compute_volume(multimesh, cell_status_multimesh);
  std::cout << "Done computing volumes with multimesh" << std::endl;
  
  ExactKernel::FT cgal_volume = 0.;
  double multimesh_volume = 0.;

  //dolfin_assert(cell_status_cgal.size() == cell_status_multimesh.size());
  for (std::size_t i = 0; i < cell_status_multimesh.size(); i++)
  {
    const std::vector<std::pair<CELL_STATUS, ExactKernel::FT> >& current_cgal = cell_status_cgal[i];
    const std::vector<std::pair<CELL_STATUS, double> >& current_multimesh = cell_status_multimesh[i];

    dolfin_assert(current_cgal.size() == current_multimesh.size());
      
    std::cout << "Cells in part " << i << ": " << std::endl;
    for (std::size_t j = 0; j < current_multimesh.size(); j++)
    {
      std::cout << "  Cell " << j << std::endl;
      std::cout << "    Multimesh: " << cell_status_str(current_multimesh[j].first) << " (" << current_multimesh[j].second << ")" << std::endl;
      std::cout << "    CGAL:      " << cell_status_str(current_cgal[j].first) << " (" << current_cgal[j].second << ")" << std::endl;
      std::cout << "      Diff:    " << CGAL::abs(current_cgal[j].second - current_multimesh[j].second) << std::endl;
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
  std::cout << "CGAL:      " << cgal_volume << ", error: " << CGAL::abs(exact_volume-cgal_volume) << std::endl;
}

int main(int argc, char** argv)
{
  test_multiple_meshes_with_rotation();
}
