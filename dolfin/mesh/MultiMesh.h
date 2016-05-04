// Copyright (C) 2014-2016 Anders Logg
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
// First added:  2014-03-03
// Last changed: 2016-03-02

#ifndef __MULTI_MESH_H
#define __MULTI_MESH_H

#include <memory>
#include <vector>
#include <map>
#include <deque>

#include <dolfin/plot/plot.h>
#include <dolfin/common/Variable.h>
#include <dolfin/geometry/Point.h>

namespace dolfin
{

  // Forward declarations
  class Cell;
  class Mesh;
  class BoundaryMesh;
  class BoundingBoxTree;

  // Typedefs
  typedef std::pair<std::vector<double>, std::vector<double> > quadrature_rule;

  // FIXME August:
  typedef std::vector<Point> Simplex;
  typedef std::vector<Simplex> Polyhedron;
  typedef std::vector<std::size_t> IncExcKey;

  /// This class represents a collection of meshes with arbitrary
  /// overlaps. A multimesh may be created from a set of standard
  /// meshes spaces by repeatedly calling add(), followed by a call to
  /// build(). Note that a multimesh is not useful until build() has
  /// been called.

  class MultiMesh : public Variable
  {
  public:

    /// Create empty multimesh
    MultiMesh();

    /// Create multimesh from given list of meshes
    MultiMesh(std::vector<std::shared_ptr<const Mesh>> meshes,
              std::size_t quadrature_order);

    //--- Convenience constructors ---

    /// Create multimesh from one mesh
    MultiMesh(std::shared_ptr<const Mesh> mesh_0,
              std::size_t quadrature_order);

    /// Create multimesh from two meshes
    MultiMesh(std::shared_ptr<const Mesh> mesh_0,
              std::shared_ptr<const Mesh> mesh_1,
              std::size_t quadrature_order);

    /// Create multimesh from three meshes
    MultiMesh(std::shared_ptr<const Mesh> mesh_0,
              std::shared_ptr<const Mesh> mesh_1,
              std::shared_ptr<const Mesh> mesh_2,
              std::size_t quadrature_order);

    /// Destructor
    ~MultiMesh();

    /// Return the number of meshes (parts) of the multimesh
    ///
    /// *Returns*
    ///     std::size_t
    ///         The number of meshes (parts) of the multimesh.
    std::size_t num_parts() const;

    /// Return mesh (part) number i
    ///
    /// *Arguments*
    ///     i (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     _Mesh_
    ///         Mesh (part) number i
    std::shared_ptr<const Mesh> part(std::size_t i) const;

    /// Return the list of uncut cells for given part. The uncut cells
    /// are defined as all cells that don't collide with any cells in
    /// any other part with higher part number.
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         List of uncut cell indices for given part
    const std::vector<unsigned int>& uncut_cells(std::size_t part) const;

    /// Return the list of cut cells for given part. The cut cells are
    /// defined as all cells that collide with the boundary of any
    /// part with higher part number.
    ///
    /// FIXME: Figure out whether this makes sense; a cell may collide
    /// with the boundary of part j but may still be covered
    /// completely by the domain of part j + 1. Possible solution is
    /// to for each part i check overlapping parts starting from the
    /// top and working back down to i + 1.
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         List of cut cell indices for given part
    const std::vector<unsigned int>& cut_cells(std::size_t part) const;

    /// Return the list of covered cells for given part. The covered
    /// cells are defined as all cells that collide with the domain of
    /// any part with higher part number, but not with the boundary of
    /// that part; in other words cells that are completely covered by
    /// any other part (and which therefore are inactive).
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         List of covered cell indices for given part
    const std::vector<unsigned int>& covered_cells(std::size_t part) const;

    /// Return the collision map for cut cells of the given part
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::map<unsigned int, std::vector<std::pair<std::size_t, unsigned int> > >
    ///         A map from cell indices of cut cells to a list of
    ///         cutting cells. Each cutting cell is represented as a
    ///         pair (part_number, cutting_cell_index).
    const std::map<unsigned int,
                   std::vector<std::pair<std::size_t, unsigned int> > >&
    collision_map_cut_cells(std::size_t part) const;

    /// Return quadrature rules for cut cells on the given part
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::map<unsigned int, std::pair<std::vector<double>, std::vector<double> > >
    ///         A map from cell indices of cut cells to quadrature
    ///         rules. Each quadrature rule is represented as a pair
    ///         of a flattened array of quadrature points and a
    ///         corresponding array of quadrature weights.
    const std::map<unsigned int, quadrature_rule >&
    quadrature_rule_cut_cells(std::size_t part) const;

    /// Return quadrature rule for a given cut cell on the given part
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///     cell (unsigned int)
    ///         The cell index
    ///
    /// *Returns*
    ///     std::pair<std::vector<double>, std::vector<double> >
    ///         A quadrature rule represented as a pair of a flattened
    ///         array of quadrature points and a corresponding array
    ///         of quadrature weights. An error is raised if the given
    ///         cell is not in the map.
    ///
    /// Developer note: this function is mainly useful from Python and
    /// could be replaced by a suitable typemap that would make the
    /// previous more general function accessible from Python.
    quadrature_rule
    quadrature_rule_cut_cell(std::size_t part, unsigned int cell_index) const;

    /// Return quadrature rules for the overlap on the given part.
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::map<unsigned int, std::pair<std::vector<double>, std::vector<double> > >
    ///         A map from cell indices of cut cells to quadrature
    ///         rules.  A separate quadrature rule is given for each
    ///         cutting cell and stored in the same order as in the
    ///         collision map. Each quadrature rule is represented as
    ///         a pair of an array of quadrature points and a
    ///         corresponding flattened array of quadrature weights.
    const std::map<unsigned int, std::vector<quadrature_rule> >&
    quadrature_rule_overlap(std::size_t part) const;

    /// Return quadrature rules for the interface on the given part
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::map<unsigned int, std::pair<std::vector<double>, std::vector<double> > >
    ///         A map from cell indices of cut cells to quadrature
    ///         rules on an interface part cutting through the cell.
    ///         A separate quadrature rule is given for each cutting
    ///         cell and stored in the same order as in the collision
    ///         map. Each quadrature rule is represented as a pair of
    ///         an array of quadrature points and a corresponding
    ///         flattened array of quadrature weights.
    const std::map<unsigned int, std::vector<quadrature_rule> >&
    quadrature_rule_interface(std::size_t part) const;

    /// Return facet normals for the interface on the given part
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::map<unsigned int, std::vector<std::vector<double> > >
    ///         A map from cell indices of cut cells to facet normals
    ///         on an interface part cutting through the cell. A
    ///         separate list of facet normals, one for each
    ///         quadrature point, is given for each cutting cell and
    ///         stored in the same order as in the collision map. The
    ///         facet normals for each set of quadrature points is
    ///         stored as a contiguous flattened array, the length of
    ///         which should be equal to the number of quadrature
    ///         points multiplied by the geometric dimension. Puh!
    const std::map<unsigned int, std::vector<std::vector<double> > >&
    facet_normals(std::size_t part) const;

    /// Return the bounding box tree for the mesh of the given part
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::shared_ptr<const BoundingBoxTree>
    ///         The bounding box tree
    std::shared_ptr<const BoundingBoxTree>
    bounding_box_tree(std::size_t part) const;

    /// Return the bounding box tree for the boundary mesh of the
    /// given part
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::shared_ptr<const BoundingBoxTree>
    ///         The bounding box tree
    std::shared_ptr<const BoundingBoxTree>
    bounding_box_tree_boundary(std::size_t part) const;

    /// Add mesh
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh
    void add(std::shared_ptr<const Mesh> mesh);

    /// Build multimesh
    void build(std::size_t quadrature_order=2);

    /// Clear multimesh
    void clear();

    // Create matplotlib string to plot 2D multimesh
    // Only suitable for smaller meshes
    std::string plot_matplotlib(double delta_z=1) const;

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("multimesh");

      p.add("quadrature_order", 1);

      return p;
    }

  private:

    // Friend (in plot.h)
    friend void plot(std::shared_ptr<const MultiMesh>);

    // List of meshes
    std::vector<std::shared_ptr<const Mesh> > _meshes;

    // List of boundary meshes
    std::vector<std::shared_ptr<BoundaryMesh> > _boundary_meshes;

    // List of bounding box trees for meshes
    std::vector<std::shared_ptr<BoundingBoxTree> > _trees;

    // List of bounding box trees for boundary meshes
    std::vector<std::shared_ptr<BoundingBoxTree> > _boundary_trees;

    // Cell indices for all uncut cells for all parts. Access data by
    //
    //     c = _uncut_cells[i][j]
    //
    // where
    //
    //     c = cell index for an uncut cell
    //     i = the part (mesh) number
    //     j = the cell number (in the list of uncut cells)
    std::vector<std::vector<unsigned int> > _uncut_cells;

    // Cell indices for all cut cells for all parts. Access data by
    //
    //     c = _cut_cells[i][j]
    //
    // where
    //
    //     c = cell index for a cut cell
    //     i = the part (mesh) number
    //     j = the cell number (in the list of cut cells)
    std::vector<std::vector<unsigned int> > _cut_cells;

    // Cell indices for all covered cells for all parts. Access data by
    //
    //     c = _covered_cells[i][j]
    //
    // where
    //
    //     c = cell index for a covered cell
    //     i = the part (mesh) number
    //     j = the cell number (in the list of covered cells)
    std::vector<std::vector<unsigned int> > _covered_cells;

    // Developer note 1: The data structures _collision_map_cut_cells
    // and _quadrature_rules_cut_cells may be changed from maps to
    // vectors and indexed by the number of the cut cell (in the list
    // of cut cells), instead of indexed by the local cell index as
    // here, if we find that this is important for performance.
    //
    // Developer note 2: Quadrature points are naturally a part of a
    // form (or a term in a form) and not a part of a mesh. However,
    // for now we use a global (to the multimesh) quadrature rule for
    // all cut cells, for simplicity.

    // Collision map for cut cells. Access data by
    //
    //     c = _collision_map_cut_cells[i][j][k]
    //
    // where
    //
    //     c.first  = part number for the cutting mesh
    //     c.second = cell index for the cutting cell
    //            i = the part (mesh) number
    //            j = the cell number (local cell index
    //            k = the collision number (in the list of cutting cells)
    std::vector<std::map<unsigned int,
                         std::vector<std::pair<std::size_t, unsigned int> > > >
    _collision_maps_cut_cells;

    // FIXME: test saving collision with boundary in its own data
    // structure (this saves only the boundary part)
    std::vector<std::map<unsigned int,
                         std::vector<std::pair<std::size_t, unsigned int> > > >
    _collision_maps_cut_cells_boundary;

    // Quadrature rules for cut cells. Access data by
    //
    //     q = _quadrature_rules_cut_cells[i][j]
    //
    // where
    //
    //     q.first  = quadrature weights, array of length num_points
    //     q.second = quadrature points, flattened num_points x gdim array
    //            i = the part (mesh) number
    //            j = the cell number (local cell index)
    std::vector<std::map<unsigned int, quadrature_rule> >
    _quadrature_rules_cut_cells;

    // Quadrature rules for overlap. Access data by
    //
    //     q = _quadrature_rules_overlap[i][j][k]
    //
    // where
    //
    //     q.first  = quadrature weights, array of length num_points
    //     q.second = quadrature points, flattened num_points x gdim array
    //            i = the part (mesh) number
    //            j = the cell number (local cell index)
    //            k = the collision number (in the list of cutting cells)
    std::vector<std::map<unsigned int, std::vector<quadrature_rule> > >
    _quadrature_rules_overlap;

    // Quadrature rules for interface. Access data by
    //
    //     q = _quadrature_rules_interface[i][j][k]
    //
    // where
    //
    //     q.first  = quadrature weights, array of length num_points
    //     q.second = quadrature points, flattened num_points x gdim array
    //            i = the part (mesh) number
    //            j = the cell number (local cell index)
    //            k = the collision number (in the list of cutting cells)
    std::vector<std::map<unsigned int, std::vector<quadrature_rule> > >
    _quadrature_rules_interface;

    // Facet normals for interface. Access data by
    //
    //     n = _facet_normals_interface[i][j][k][
    //
    // where
    //
    //     n = a flattened array vector of facet normals, one point for
    //         each quadrature point
    //     i = the part (mesh) number
    //     j = the cell number (local cell index)
    //     k = the collision number (in the list of cutting cells)
    std::vector<std::map<unsigned int, std::vector<std::vector<double> > > >
    _facet_normals;

    // Build boundary meshes
    void _build_boundary_meshes();

    // Build bounding box trees
    void _build_bounding_box_trees();

    // Build collision maps
    void _build_collision_maps();
    //void _build_collision_maps_same_topology();
    //void _build_collision_maps_different_topology();

    // Build quadrature rules for the cut cells
    void _build_quadrature_rules_cut_cells(std::size_t quadrature_order);

    // Build quadrature rules for the overlap
    void _build_quadrature_rules_overlap(std::size_t quadrature_order);

    // FIXME
    void _build_quadrature_rules_interface(std::size_t quadrature_order);

    // Add quadrature rule for simplices in the triangulation
    // array. Returns the number of points generated for each simplex.
    std::vector<std::size_t>
    _add_quadrature_rule(quadrature_rule& qr,
                         const std::vector<double>& triangulation,
                         std::size_t tdim,
                         std::size_t gdim,
                         std::size_t quadrature_order,
                         double factor) const;

    // FIXME: since IntersectionTriangulation uses mostly std::vector<Point> create this function while fixing the interface to all functions
    std::vector<std::size_t>
      _add_quadrature_rule(quadrature_rule& qr,
			   const std::vector<Point>& triangulation,
			   std::size_t tdim,
			   std::size_t gdim,
			   std::size_t quadrature_order,
			   double factor) const
      {
	std::vector<double> flat((tdim+1)*gdim);
	for (std::size_t i = 0; i < triangulation.size(); ++i)
	  for (std::size_t d = 0; d < gdim; ++d)
	    flat[i*gdim+d] = triangulation[i][d];
	return _add_quadrature_rule(qr, flat, tdim, gdim, quadrature_order, factor);
      }

    // Add quadrature rule to existing quadrature rule (append dqr to
    // qr). Returns number of points added.
    std::size_t _add_quadrature_rule(quadrature_rule& qr,
                                     const quadrature_rule& dqr,
                                     std::size_t gdim,
                                     double factor) const;

    // Append normal to list of normals npts times
    void _add_normal(std::vector<double>& normals,
                     const Point& normal,
                     std::size_t npts,
                     std::size_t gdim) const;

    // Plot multimesh
    void _plot() const;

    // FIXME: Helper function to convert between flat triangulation
    // and list of Points (this function should not be needed: fix the
    // interface of IntersectionTriangulation instead)
    static Simplex convert(const double* x,
			   std::size_t tdim,
			   std::size_t gdim)
    {
      Simplex s(tdim + 1);
      for (std::size_t t = 0; t < tdim + 1; ++t)
	for (std::size_t d = 0; d < gdim; ++d)
	  s[t][d] = x[gdim*t + d];
      return s;
    }

    // FIXME: Helper function to convert between flat triangulation
    // and list of Points (this function should not be needed: fix the
    // interface of IntersectionTriangulation instead)
    static std::vector<Simplex> convert(const std::vector<double>& triangulation,
					std::size_t tdim,
					std::size_t gdim)
    {
      const std::size_t offset = (tdim + 1)*gdim;
      const std::size_t N = triangulation.size() / offset;
      std::vector<Simplex> simplices(N);

      for (std::size_t k = 0; k < N; ++k)
      {
	const double* x = triangulation.data() + k*offset;
	simplices[k] = convert(x, tdim, gdim);
      }

      return simplices;
    }

    // FIXME: Helper function to convert between flat triangulation
    // and list of Points (this function should not be needed: fix the
    // interface of IntersectionTriangulation instead)
    static std::vector<double> convert(const Simplex& simplex,
				       std::size_t tdim,
				       std::size_t gdim)
    {
      std::vector<double> x((tdim + 1)*gdim);

      for (std::size_t i = 0; i < tdim + 1; ++i)
	for (std::size_t d = 0; d < gdim; ++d)
	  x[i*gdim + d] = simplex[i][d];

      return x;
    }


    // FIXME: Helper function to convert between flat triangulation
    // and list of Points (this function should not be needed: fix the
    // interface of IntersectionTriangulation instead)
    static std::vector<double> convert(const std::vector<Simplex>& simplices,
				       std::size_t tdim,
				       std::size_t gdim)
    {
      const std::size_t offset = (tdim + 1)*gdim;
      const std::size_t N = simplices.size();
      std::vector<double> triangulation(N*offset);

      for (std::size_t i = 0; i < N; ++i)
      {
	const std::vector<double> x = convert(simplices[i], tdim, gdim);
	for (std::size_t j = 0; j < x.size(); ++j)
	  triangulation[i*offset + j] = x[j];
      }

      return triangulation;
    }


    // FIXME:
    //double minimum_angle(double* a, double* b, double* c) const;
    //double minimum_angle(const Simplex& s) const;
    //bool maximize_minimum_angle(Polyhedron& p) const;


  };



}


#endif
