// Copyright (C) 2009 Andre Massing
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
// Modified by Johannes Ring, 2009.
//
// First added:  2009-09-11
// Last changed: 2011-11-11

#ifndef __INTERSECTIONOPERATORIMPLEMENTATION_H
#define __INTERSECTIONOPERATORIMPLEMENTATION_H

#include <vector>
#include <utility>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/optional.hpp>

#include <dolfin/common/types.h>
#include <dolfin/mesh/Point.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/SubsetIterator.h>

#ifdef HAS_CGAL

#include "cgal_includes.h"
typedef CGAL::Simple_cartesian<double> SCK;
typedef CGAL::Exact_predicates_inexact_constructions_kernel EPICK;

namespace dolfin
{

  /// @brief Interface class for the actual implementation of the intersection operations.
  ///
  /// @internal This is neccessary since the search tree has a dimension dependent type, hence encapsulates in the
  /// inheriting IntersectionOperatorImplementation_d! It provides the glue level between the dimension independent implementation
  /// of the mesh class and the dimension dependent search structures in CGAL.
  class IntersectionOperatorImplementation
  {
  public:

    // Only default constructor, since the search tree has a dimension dependent type, hence encapsulates in the
    // inheriting IntersectionOperatorImplementation_d!

    virtual void all_intersected_entities(const Point & point, std::set<uint> & ids_result) const = 0;
    virtual void all_intersected_entities(const std::vector<Point> & points, std::set<uint> & ids_result) const = 0;

    virtual void all_intersected_entities(const MeshEntity & entity, std::vector<uint> & ids_result) const = 0;
    virtual void all_intersected_entities(const std::vector<MeshEntity> & entities, std::set<uint> & ids_result) const = 0;

    virtual void all_intersected_entities(const Mesh & another_mesh, std::set<uint> & ids_result) const = 0;
    virtual int any_intersected_entity(const Point & point) const = 0;
    virtual Point closest_point(const Point & point) const = 0;
    virtual dolfin::uint closest_cell(const Point & point) const = 0;
    virtual std::pair<Point,uint> closest_point_and_cell(const Point & point) const = 0;
    virtual double distance(const Point & point) const = 0;

  };

  /// Class which provides the dimensional implementation of the search structure
  /// for the mesh.
  template <class Primitive, class Kernel>
  class IntersectionOperatorImplementation_d : public IntersectionOperatorImplementation
  {
    typedef PrimitiveTraits<Primitive,Kernel> PT;
    typedef typename PT::K K;
    typedef MeshPrimitive<PT> CellPrimitive;

    typedef CGAL::AABB_traits<K,CellPrimitive> AABB_PrimitiveTraits;
    typedef CGAL::AABB_tree<AABB_PrimitiveTraits> Tree;

  public:

    /// Constructor
    IntersectionOperatorImplementation_d(boost::shared_ptr<const Mesh> mesh)
      : point_search_tree_constructed(false)
    {
      build_tree(*mesh);
    }

    IntersectionOperatorImplementation_d(const MeshFunction<uint> labels, uint label)
    : point_search_tree_constructed(false)
    {
      // Build CGAL AABB tree
      build_tree(labels, label);
    }

    virtual void all_intersected_entities(const Point& point, std::set<uint>& ids_result) const;
    virtual void all_intersected_entities(const std::vector<Point>& points, std::set<uint>& ids_result) const;

    virtual void all_intersected_entities(const MeshEntity& entity, std::vector<uint>& ids_result) const;
    virtual void all_intersected_entities(const std::vector<MeshEntity>& entities, std::set<uint>& ids_result) const;

    virtual void all_intersected_entities(const Mesh& another_mesh, std::set<uint>& ids_result) const;

    virtual  int any_intersected_entity(const Point& point) const;

    virtual Point closest_point(const Point& point) const;
    virtual dolfin::uint closest_cell(const Point& point) const;
    virtual std::pair<Point, dolfin::uint> closest_point_and_cell(const Point& point) const;
    virtual double distance(const Point & point) const;

    ///Topological dimension of the mesh.
    static const uint dim = PT::dim;

  private:

    /// Build AABB_tree search tree
    void build_tree(const Mesh& mesh);

    /// Build AABB_tree search tree using selected entities
    void build_tree(const MeshFunction<uint>& labels, uint label);

    /// The AABB search tree
    boost::scoped_ptr<Tree> tree;

    /// Boolean flag to indicate whether Kd tree has already been built 
    mutable bool point_search_tree_constructed;

  };

  template <class P, class K>
  void IntersectionOperatorImplementation_d<P, K>::all_intersected_entities(const Point& point, std::set<uint>& ids_result) const
  {
    std::insert_iterator< std::set<uint> > output_it(ids_result, ids_result.end());
    tree->all_intersected_primitives(PrimitiveTraits<PointPrimitive,K>::datum(point), output_it);
  }

  template <class P, class K>
  void IntersectionOperatorImplementation_d<P, K>::all_intersected_entities(const std::vector<Point>& points, std::set<uint>& ids_result) const
  {
    std::insert_iterator< std::set<uint> > output_it(ids_result, ids_result.end());
    for (std::vector<Point>::const_iterator p = points.begin(); p != points.end(); ++p)
    {
      tree->all_intersected_primitives(PrimitiveTraits<PointPrimitive,K>::datum(*p), output_it);
    }
  }

  template<class P, class K>
  void IntersectionOperatorImplementation_d<P, K>::all_intersected_entities(const MeshEntity& entity, std::vector<uint>& ids_result) const
  {
    std::insert_iterator< std::vector<uint> > output_it(ids_result, ids_result.end());
    //Convert entity to corresponding cgal geomtric object according to the mesh
    //entity dimension.
    switch (entity.dim())
    {
      case 0: tree->all_intersected_primitives(PrimitiveTraits<PointCell,K>::datum(entity), output_it); break;
      case 1: tree->all_intersected_primitives(PrimitiveTraits<IntervalCell,K>::datum(entity), output_it); break;
      case 2: tree->all_intersected_primitives(PrimitiveTraits<TriangleCell,K>::datum(entity), output_it); break;
      case 3: tree->all_intersected_primitives(PrimitiveTraits<TetrahedronCell,K>::datum(entity), output_it); break;
      default: error("DOLFIN IntersectionOperatorImplementation::all_intersected_entities: \n Mesh CellType is not known.");
    }
  }

  template<class P, class K>
  void IntersectionOperatorImplementation_d<P, K>::all_intersected_entities(const std::vector<MeshEntity>& entities, std::set<uint>& ids_result) const
  {
    std::insert_iterator< std::set<uint> > output_it(ids_result, ids_result.end());
    for (std::vector<MeshEntity>::const_iterator entity = entities.begin(); entity != entities.end(); ++entity)
      switch(entity->dim())
      {
        case 0:
          tree->all_intersected_primitives(PrimitiveTraits<PointCell,K>::datum(*entity), output_it); break;
        case 1:
          tree->all_intersected_primitives(PrimitiveTraits<IntervalCell,K>::datum(*entity), output_it); break;
        case 2:
          tree->all_intersected_primitives(PrimitiveTraits<TriangleCell,K>::datum(*entity), output_it); break;
        case 3:
          tree->all_intersected_primitives(PrimitiveTraits<TetrahedronCell,K>::datum(*entity), output_it); break;
        default:  error("DOLFIN IntersectionOperatorImplementation::all_intersected_entities: \n Mesh EntityType is not known.");
      }
  }

  template<class P, class K>
  void IntersectionOperatorImplementation_d<P, K>::all_intersected_entities(const Mesh& another_mesh, std::set<uint>& ids_result) const
  {
    //Avoid instantiation of an insert_iterator for each cell.
    std::insert_iterator<std::set<uint> > output_it(ids_result, ids_result.end());
    switch( another_mesh.type().cell_type())
    {
      case CellType::point:
        for (CellIterator cell(another_mesh); !cell.end(); ++cell)
          tree->all_intersected_primitives(PrimitiveTraits<PointCell,K>::datum(*cell), output_it);
        break;
      case CellType::interval:
        if (dim == 1 || dim == 3)
          dolfin_not_implemented();
        else
          for (CellIterator cell(another_mesh); !cell.end(); ++cell)
            tree->all_intersected_primitives(PrimitiveTraits<IntervalCell,K>::datum(*cell), output_it);
          break;
      case CellType::triangle:
        for (CellIterator cell(another_mesh); !cell.end(); ++cell)
          tree->all_intersected_primitives(PrimitiveTraits<TriangleCell,K>::datum(*cell), output_it);
        break;
      case CellType::tetrahedron:
          for (CellIterator cell(another_mesh); !cell.end(); ++cell)
            tree->all_intersected_primitives(PrimitiveTraits<TetrahedronCell,K>::datum(*cell), output_it);
          break;
      default:
        error("DOLFIN IntersectionOperatorImplementation::all_intersected_entities: \n Mesh CellType is not known.");
    }
  }

  template <class P, class K>
  int IntersectionOperatorImplementation_d<P, K>::any_intersected_entity(const Point& point) const
  {
    boost::optional<uint> id = tree->any_intersected_primitive(PrimitiveTraits<PointPrimitive,K>::datum(point));
    if (id)
      return *id;
    else
      return -1;
  }

  ///Temporary ugly helper class to specialize for non existing implementation for Tetrahedron meshes.
  template<class P, class K, class Tree>
  struct ClosestPoint
  {
    typedef typename K::Point_3 Point_3;

    static Point_3 compute(const Tree& tree, const Point_3& point)
    {
      return tree.closest_point(point);
    }
  };

  // Partial special for 3D since the nearest_point_3 which is internally used in CGAL can not yet handles tetrahedrons.
  // Have to supply myself :)
  template<class K, class Tree>
  struct ClosestPoint<TetrahedronCell, K, Tree>
  {
    typedef typename K::Point_3 Point_3;

    static Point_3 compute(const Tree& tree, const Point_3& point)
    {
      dolfin_not_implemented();
      return Point_3();
    }
  };

  // Partial special for 3D since the nearest_point_3 which is internally used in CGAL can not yet handles tetrahedrons.
  // Have to supply myself :)
  template<class K, class Tree>
  struct ClosestPoint<PointCell, K, Tree>
  {
    typedef typename K::Point_3 Point_3;

    static Point_3 compute(const Tree& tree, const Point_3& point)
    {
      dolfin_not_implemented();
      return Point_3();
    }
  };

  template<class P, class K, class Tree>
  struct ClosestPointAndPrimitive
  {
    typedef typename K::Point_3 Point_3;
    typedef typename Tree::Point_and_primitive_id Point_and_primitive_id;
    static std::pair<Point,dolfin::uint> compute(const Tree& tree, const Point_3& point)
    {
       Point_and_primitive_id pp = tree.closest_point_and_primitive(point);
       return std::pair<Point,uint>(Point(pp.first), pp.second);
    }
  };

  // Partial special for 3D since the nearest_point_3 which is internally used in CGAL can not yet handles tetrahedrons.
  // Have to supply myself :)
  template<class K, class Tree>
  struct ClosestPointAndPrimitive<TetrahedronCell, K, Tree>
  {
    typedef typename K::Point_3 Point_3;

    static std::pair<Point, dolfin::uint> compute(const Tree& tree, const Point_3& point)
    {
      dolfin_not_implemented();
      return std::pair<Point,uint>(Point(), 0);
    }
  };

  // Partial special for 3D since the nearest_point_3 which is internally used in CGAL can not yet handles *points*.
  // Have to supply myself :) THAT should not be difficult...
  template<class K, class Tree>
  struct ClosestPointAndPrimitive<PointCell, K, Tree>
  {
    typedef typename K::Point_3 Point_3;

    static std::pair<Point,dolfin::uint> compute(const Tree& tree, const Point_3& point)
    {
      dolfin_not_implemented();
      return std::pair<Point,uint>(Point(), 0);
    }
  };

  template <class P, class K>
  Point IntersectionOperatorImplementation_d<P, K>::closest_point(const Point& point) const
  {
    if (!point_search_tree_constructed)
      point_search_tree_constructed = tree->accelerate_distance_queries();
    return  Point(ClosestPoint<P,K,Tree>::compute(*tree,PrimitiveTraits<PointPrimitive,K>::datum(point)));
  }

  template <class P, class K>
  dolfin::uint IntersectionOperatorImplementation_d<P, K>::closest_cell(const Point& point) const
  {
    return closest_point_and_cell(point).second;
  }
   
  ///Temporary ugly helper class to specialize for non existing implementation for Tetrahedron meshes.
  template<class P, class K, class Tree>
  struct Distance
  {
    typedef typename K::Point_3 Point_3;

    static double compute(const Tree& tree, const Point_3& point)
    {
      return std::sqrt(tree.squared_distance(point));
    }
  };

  // Partial special for 3D since the nearest_point_3 which is internally used in CGAL can not yet handles tetrahedrons.
  // Have to supply myself :)
  template<class K, class Tree>
  struct Distance<TetrahedronCell, K, Tree>
  {
    typedef typename K::Point_3 Point_3;

    static double compute(const Tree& tree, const Point_3& point)
    {
      dolfin_not_implemented();
      return 0;
    }
  };

  // Partial special for 3D since the nearest_point_3 which is internally used in CGAL can not yet handles *points*.
  // Have to supply myself :) THAT should not be difficult...
  template<class K, class Tree>
  struct Distance<PointCell, K, Tree>
  {
    typedef typename K::Point_3 Point_3;

    static double compute(const Tree& tree, const Point_3& point)
    {
      dolfin_not_implemented();
      return 0;
    }
  };

  template <class P, class K>
  double IntersectionOperatorImplementation_d<P, K>::distance(const Point & point) const
  {
    if (!point_search_tree_constructed)
      point_search_tree_constructed = tree->accelerate_distance_queries();
    return  Distance<P,K,Tree>::compute(*tree,PrimitiveTraits<PointPrimitive,K>::datum(point));
  }
    
  template <class P, class K>
  std::pair<Point,uint> IntersectionOperatorImplementation_d<P, K>::closest_point_and_cell(const Point& point) const
  {
    if (!point_search_tree_constructed)
      point_search_tree_constructed = tree->accelerate_distance_queries();
    return ClosestPointAndPrimitive<P,K,Tree>::compute(*tree,PrimitiveTraits<PointPrimitive,K>::datum(point));
  }

  template <class P, class K>
  void IntersectionOperatorImplementation_d<P, K>::build_tree(const Mesh & mesh)
  {
    MeshEntityIterator entity_iter(mesh, mesh.topology().dim());
    tree.reset(new Tree(entity_iter,entity_iter.end_iterator()));
    point_search_tree_constructed = false;
  }

  template <class P, class K>
  void IntersectionOperatorImplementation_d<P, K>::build_tree(const MeshFunction<uint> & labels, uint label)
  {
    SubsetIterator entity_iter(labels, label);
    tree.reset(new Tree(entity_iter,entity_iter.end_iterator()));
    point_search_tree_constructed = false;
  }
}

#else

  // Fake interface to allow creation of an IntersectionOperator instance
  // *without* functionality.  IntersectionOperator uses lazy initialization.
  // Throw an exception  if a IntersectionOperatorImplementation instance should
  // be created without CGAL support.

namespace dolfin
{

  class IntersectionOperatorImplementation
  {
  public:

    IntersectionOperatorImplementation()
    {
      error("DOLFIN has been compiled without CGAL, IntersectionOperatorImplementation is not available.");
    }
    virtual ~IntersectionOperatorImplementation() {}
    virtual void all_intersected_entities(const Point& point, std::set<uint>& ids_result) const {}
    virtual void all_intersected_entities(const std::vector<Point>& points, std::set<uint>& ids_result) const {}
    virtual void all_intersected_entities(const MeshEntity& entity, std::vector<uint>& ids_result) const {};
    virtual void all_intersected_entities(const std::vector<MeshEntity>& entities, std::set<uint>& ids_result) const {};
    virtual void all_intersected_entities(const Mesh& another_mesh, std::set<uint>& ids_result) const {}
    virtual int any_intersected_entity(const Point& point) const { return -1; }
    virtual Point closest_point(const Point& point) const { return Point(); }
    virtual dolfin::uint closest_cell(const Point& point) const { return 0; }
    virtual std::pair<Point,uint> closest_point_and_cell(const Point& point) const { return std::pair<Point, uint>(); }
    virtual double distance(const Point & point) const { return 0; }

  };
}

#endif

#endif
