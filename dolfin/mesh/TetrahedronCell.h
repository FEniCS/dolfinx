// Copyright (C) 2006-2013 Anders Logg
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
// Modified by Johan Hoffman, 2006.
// Modified by Garth N. Wells, 2006.
// Modified by Kristoffer Selim, 2008.
//
// First added:  2006-06-05
// Last changed: 2014-01-31

#ifndef __TETRAHEDRON_CELL_H
#define __TETRAHEDRON_CELL_H

#include <vector>
#include "CellType.h"

namespace dolfin
{

  class Cell;

  /// This class implements functionality for tetrahedral meshes.

  class TetrahedronCell : public CellType
  {
  public:

    /// Specify cell type and facet type
    TetrahedronCell() : CellType(tetrahedron, triangle) {}

    /// Return topological dimension of cell
    std::size_t dim() const;

    /// Return number of entitites of given topological dimension
    std::size_t num_entities(std::size_t dim) const;

    /// Return number of vertices for entity of given topological dimension
    std::size_t num_vertices(std::size_t dim) const;

    /// Return orientation of the cell
    std::size_t orientation(const Cell& cell) const;

    /// Create entities e of given topological dimension from vertices v
    void create_entities(std::vector<std::vector<unsigned int> >& e,
                         std::size_t dim, const unsigned int* v) const;

    /// Regular refinement of cell
    void refine_cell(Cell& cell, MeshEditor& editor,
                     std::size_t& current_cell) const;

    /// Irregular refinement of cell
    void refine_cellIrregular(Cell& cell, MeshEditor& editor,
                              std::size_t& current_cell, std::size_t refinement_rule,
                              std::size_t* marked_edges) const;

    /// Compute volume of tetrahedron
    double volume(const MeshEntity& tetrahedron) const;

    /// Compute diameter of tetrahedron
    double diameter(const MeshEntity& tetrahedron) const;

    /// Compute squared distance to given point
    double squared_distance(const Cell& cell, const Point& point) const;

    /// Compute component i of normal of given facet with respect to the cell
    double normal(const Cell& cell, std::size_t facet, std::size_t i) const;

    /// Compute normal of given facet with respect to the cell
    Point normal(const Cell& cell, std::size_t facet) const;

    /// Compute normal to given cell (viewed as embedded in 4D ...)
    Point cell_normal(const Cell& cell) const;

    /// Compute the area/length of given facet with respect to the cell
    double facet_area(const Cell& cell, std::size_t facet) const;

    /// Order entities locally
    void order(Cell& cell,
               const std::vector<std::size_t>& local_to_global_vertex_indices) const;

    /// Check whether given point collides with cell
    bool collides(const Cell& cell, const Point& point) const;

    /// Check whether given entity collides with cell
    bool collides(const Cell& cell, const MeshEntity& entity) const;

    /// Compute triangulation of intersection of two cells
    virtual std::vector<double>
      triangulate_intersection(const Cell& c0, const Cell& c1) const;

    /// Return description of cell type
    std::string description(bool plural) const;

  private:

    // Find local index of edge i according to ordering convention
    std::size_t find_edge(std::size_t i, const Cell& cell) const;

    // Check whether point is outside region defined by facet ABC.
    // The fourth vertex is needed to define the orientation.
    bool point_outside_of_plane(const Point& point,
                                const Point& A,
                                const Point& B,
                                const Point& C,
                                const Point& D) const;


    /// Check whether given triangle collides with cell
    bool collides_triangle(const Cell& cell, const MeshEntity& entity) const;

    /// Check whether given tetrahedron collides with cell
    bool collides_tetrahedron(const Cell& cell, const MeshEntity& entity) const;

    // Helper function for collides_tetrahedron: checks if plane pv1 is a separating plane. Stores local coordinates bc and the mask bit maskEdges.
    bool separating_plane_face_A_1(const std::vector<Point>& pv1,
				   const Point& n,
				   std::vector<double>& bc,
				   int& maskEdges) const;

    // Helper function for collides_tetrahedron: checks if plane v1,v2 is a separating plane. Stores local coordinates bc and the mask bit maskEdges.
    bool separating_plane_face_A_2(const std::vector<Point>& v1,
				   const std::vector<Point>& v2,
				   const Point& n,
				   std::vector<double>& bc,
				   int& maskEdges) const;
		
    // Helper function for collides_tetrahedron: checks if plane pv2 is a separating plane.
    bool separating_plane_face_B_1(const std::vector<Point>& P_V2,
				   const Point& n) const
    {
      return ((P_V2[0].dot(n) > 0) &&
	      (P_V2[1].dot(n) > 0) &&
	      (P_V2[2].dot(n) > 0) &&
	      (P_V2[3].dot(n) > 0));
    }

    // Helper function for collides_tetrahedron: checks if plane v1,v2 is a separating plane.  
    bool separating_plane_face_B_2(const std::vector<Point>& V1,
				   const std::vector<Point>& V2,
				   const Point& n) const
    {
      return (((V1[0]-V2[1]).dot(n) > 0) &&
	      ((V1[1]-V2[1]).dot(n) > 0) &&
	      ((V1[2]-V2[1]).dot(n) > 0) &&
	      ((V1[3]-V2[1]).dot(n) > 0));
    }
    
    // Helper function for collides_tetrahedron: checks if edge is in the plane separating faces f0 and f1. 
    bool separating_plane_edge_A(const std::vector<std::vector<double> >& Coord_1,
				 const std::vector<int>& masks,
				 int f0, 
				 int f1) const;
 
    // Helper function for triangulate_intersection: computes edge face intersection
    bool edge_face_collision(const Point& r,
			     const Point& s,
			     const Point& t,
			     const Point& a,
			     const Point& b,
			     Point& pt) const;
    
    // Helper function for triangulate_intersection: computes edge edge intersection.
    bool edge_edge_collision(const Point& a,
			     const Point& b,
			     const Point& c,
			     const Point& d,
			     Point& pt) const;


  };

}

#endif
