// Copyright (C) 2009-2011 Andre Massing
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
// First added:  2009-09-01
// Last changed: 2011-11-11

#ifndef __INTERSECTIONOPERATOR_H
#define __INTERSECTIONOPERATOR_H

#include <string>
#include <utility>
#include <vector>

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include <dolfin/common/types.h>

namespace dolfin
{

  // Forward declarations
  class MeshEntity;
  class Mesh;
  template <class T> class MeshFunction;
  class Point;
  class IntersectionOperatorImplementation;

  class IntersectionOperator
  {
  public:

    /// Create intersection detector for a given mesh 
    ///
    ///
    /// @param kernel_type The CGAL geometric kernel is used to compute predicates,
    /// intersections and such. Depending on this choice the kernel
    /// (kernel_type = "ExcactPredicates") can compute predicates excactly
    /// (without roundoff error) or only approximately (default, kernel_type =
    /// "SimpleCartesian").
    IntersectionOperator(const Mesh& _mesh,
                         const std::string& kernel_type = "SimpleCartesian");

    IntersectionOperator(boost::shared_ptr<const Mesh> _mesh,
                         const std::string& kernel_type = "SimpleCartesian");

    /// Create  IntersectionOperator for a given mesh
    ///
    /// *Arguments*
    ///     labels (_MeshFunction<unsigned int>_)
    ///         A MeshFunction over entities labeling the part of the Mesh 
    ///         for which the distance will be measured to
    ///
    ///     label (uint)
    ///         The label determining the part of the mesh for which 
    ///         the distance will be measured to
    ///
    ///     kernel_type (std::string)
    ///         The CGAL geometric kernel which is used to compute predicates,
    ///         intersections and such. Depending on this choice the kernel
    ///         (kernel_type = "ExcactPredicates") can compute predicates 
    ///         excactly (without roundoff error) or only approximately 
    ///         default value is "SimpleCartesian".
    IntersectionOperator(const MeshFunction<unsigned int>& labels, 
			 uint label, 
                         const std::string& kernel_type = "SimpleCartesian");

    /// Create IntersectionOperator for a given mesh (shared data)
    ///
    /// *Arguments*
    ///     labels (_MeshFunction<unsigned int>_)
    ///         A MeshFunction over facets labeling the part of the Boundary 
    ///         for which the distance will be measured to
    ///
    ///     label (uint)
    ///         The label determining the part of the mesh for which 
    ///         the distance will be measured to
    ///
    ///     kernel_type (std::string)
    ///         The CGAL geometric kernel which is used to compute predicates,
    ///         intersections and such. Depending on this choice the kernel
    ///         (kernel_type = "ExcactPredicates") can compute predicates 
    ///         excactly (without roundoff error) or only approximately 
    ///         default value is "SimpleCartesian".
    IntersectionOperator(boost::shared_ptr<const MeshFunction<unsigned int> > labels,
			 uint label, const std::string&
			 kernel_type="SimpleCartesian");

    /// Destructor. Needed be explicit written, otherwise default inline
    /// here, with prohibits pImpl with scoped_ptr.
    ~IntersectionOperator();

    /// Compute all id of all cells which are intersects by a \em point.
    /// \param[out] ids_result The ids of the intersected entities are saved in a set for efficienty
    /// reasons, to avoid to sort out duplicates later on.
    void all_intersected_entities(const Point & point,
                                  std::set<uint>& ids_result) const;

    /// Compute all id of all cells which are intersects any point in \em points.
    /// \param[out] ids_result The ids of the intersected entities are saved in a set for efficienty
    /// reasons, to avoid to sort out duplicates later on.
    void all_intersected_entities(const std::vector<Point>& points,
                                  std::set<uint>& ids_result) const;

    /// Compute all id of all cells which are intersects by a \em entity.
    /// \param[out] ids_result The ids of the intersected entities are saved in a vector.
    /// This allows is more efficent than using a set and allows a map between
    /// the (external) cell and the intersected cell of the mesh. If you
    /// are only interested in intersection with a list of cells without caring about which
    /// cell what intersected by which one, use
    /// void IntersectionOperator::all_intersected_entities(const std::vector<Cell> &, std::set<uint> &) const;
    /// @internal
    /// @todo This function has to improved: 1) it requires the object the
    /// mesh is to be cut with to be another mesh entitiy instead of being just a
    /// kind of geometric object. 2) Requires a runtime switch 3) would require a
    /// implementation for each geometric  primitive if they have no common base
    /// class.
    void all_intersected_entities(const MeshEntity & entity, std::vector<uint> & ids_result) const;

    /// Compute all id of all cells which are intersects by any of the entities in \em entities. This
    /// \param[out] ids_result The ids of the intersected set are saved in a set for efficienty
    /// reasons, to avoid to sort out duplicates later on.
    void all_intersected_entities(const std::vector<MeshEntity> & entities, std::set<uint> & ids_result) const;

    /// Compute all id of all cells which are intersects by the given mesh \em another_mesh;
    /// \param[out] ids_result The ids of the intersected entities are saved in a set for efficienty
    /// reasons, to avoid to sort out duplicates later on.
    void all_intersected_entities(const Mesh& another_mesh,
                                  std::set<uint>& ids_result) const;

    /// Computes only the first id of the entity, which contains the point. Returns -1 if no cell is intersected.
    /// @internal @remark This makes the function evaluation significantly faster.
    int any_intersected_entity(const Point& point) const;

    /// Computes the point inside the mesh which is closest to the point query.
    Point closest_point(const Point & point) const;

    /// Computes the index of the cell inside the mesh which are closest to the point query.
    dolfin::uint closest_cell(const Point & point) const;

    /// Computes the point inside the mesh and the corresponding cell index
    /// that are closest to the point query.
    std::pair<Point,uint> closest_point_and_cell(const Point & point) const;

    /// Computes the distance between the given point and the nearest entity
    double distance(const Point & point) const;

    /// Rebuilds the underlying search structure from scratch and uses
    /// the kernel kernel_type underlying CGAL Geometry kernel.
    void reset_kernel(const std::string& kernel_type  = "SimpleCartesian");

    /// Clears search structure. Should be used if the mesh has changed
    void clear();

    const Mesh& mesh() const;

  private:

    // Helper function to introduce lazy initialization.
    const IntersectionOperatorImplementation& rImpl() const;

    // Factory function to create the dimension dependent intersection
    // operator implementation.
    IntersectionOperatorImplementation*
        create_intersection_operator(const std::string & kernel_type);

    // Pointer to implementation. Mutable to enable lazy initialization.
    mutable boost::scoped_ptr<IntersectionOperatorImplementation> _pImpl;

    // Pointer to mesh.
    boost::shared_ptr<const Mesh> _mesh;

    // Pointer to mesh function
    boost::shared_ptr<const MeshFunction<uint> > _labels;

    // Label if MeshFunction is used
    uint _label;
    
    // Flag if MeshFunction is used
    bool _use_labels;

    // String description of the used geometry kernel.
    std::string _kernel_type;

  };

}

#endif
