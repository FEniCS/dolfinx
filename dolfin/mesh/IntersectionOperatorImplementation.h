// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johannes Ring, 2009.
//
// First added:  2009-09-11
// Last changed: 2010-03-02

#ifndef __INTERSECTIONOPERATORIMPLEMENTATION_H
#define __INTERSECTIONOPERATORIMPLEMENTATION_H


#include <vector>
#include <utility>

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/optional.hpp>

#include <dolfin/common/types.h>

#include "Point.h"
#include "Mesh.h"


#ifdef HAS_CGAL

#include "cgal_includes.h"

typedef CGAL::Simple_cartesian<double> SCK;
typedef CGAL::Exact_predicates_inexact_constructions_kernel EPICK;

namespace dolfin
{

  ///@brief Interface class for the actual implementation of the intersection operations.
  ///
  ///@internal This is neccessary since the search tree has a dimension dependent type, hence encapsulates in the 
  ///inheriting IntersectionOperatorImplementation_d! It provides the glue level between the dimension independent implementation
  ///of the mesh class and the dimension dependent search structures in CGAL.
  class IntersectionOperatorImplementation
  {
  public:

    //Only default constructor, since the search tree has a dimension dependent type, hence encapsulates in the 
    //inheriting IntersectionOperatorImplementation_d!
    
    virtual void all_intersected_entities(const Point & point, uint_set & ids_result) const = 0; 
    virtual void all_intersected_entities(const std::vector<Point> & points, uint_set & ids_result) const = 0; 

    virtual void all_intersected_entities(const MeshEntity & entity, std::vector<uint> & ids_result) const = 0;
    virtual void all_intersected_entities(const std::vector<MeshEntity> & entities, uint_set & ids_result) const = 0;

    virtual void all_intersected_entities(const Mesh & another_mesh, uint_set & ids_result) const = 0;
    virtual int any_intersected_entity(const Point & point) const = 0; 
    virtual Point closest_point(const Point & point) const = 0;
    virtual std::pair<Point,uint> closest_point_and_entity_index(const Point & point) const = 0;
  };

  ///Class which provides the dimensional implementation of the search structure
  //for the mesh.
  template <class PrimitiveTrait>
  class IntersectionOperatorImplementation_d : public IntersectionOperatorImplementation
  {
    typedef PrimitiveTrait PT;
    typedef typename PT::K K;
    typedef MeshPrimitive<PrimitiveTrait> CellPrimitive;

//    typedef Tree_Traits<K,CellPrimitive> AABB_PrimitiveTraits;
    typedef CGAL::AABB_traits<K,CellPrimitive> AABB_PrimitiveTraits;
    typedef CGAL::AABB_tree<AABB_PrimitiveTraits> Tree;
//    typedef Tree::Point_and_primitive_id Point_and_primitive_id;

  public:
    ///Constructor. 
    IntersectionOperatorImplementation_d(boost::shared_ptr<const Mesh> _mesh) 
      : _mesh(_mesh), point_search_tree_constructed(false)
    {
      build_tree();
    }

    virtual void all_intersected_entities(const Point & point, uint_set & ids_result) const; 
    virtual void all_intersected_entities(const std::vector<Point> & points, uint_set & ids_result) const;

    virtual void all_intersected_entities(const MeshEntity & entity, std::vector<uint> & ids_result) const;
    virtual void all_intersected_entities(const std::vector<MeshEntity> & entities, uint_set & ids_result) const;

    virtual void all_intersected_entities(const Mesh & another_mesh, uint_set & ids_result) const;

    virtual  int any_intersected_entity(const Point & point) const;

    virtual Point closest_point(const Point & point) const;
    virtual std::pair<Point,dolfin::uint> closest_point_and_entity_index(const Point & point) const;

    ///Topological dimension of the mesh.
    static const uint dim = PrimitiveTrait::dim;

  private:

    void build_tree();
    boost::shared_ptr<const Mesh> _mesh;
    boost::scoped_ptr<Tree> tree;
    bool point_search_tree_constructed;

  };

  template <class PT>
  void IntersectionOperatorImplementation_d<PT>::all_intersected_entities(const Point & point, uint_set & ids_result) const
  {
    //@remark For a set the start iterator required by the insert_iterator constructor does not really matter.
    std::insert_iterator< uint_set > output_it(ids_result,ids_result.end());
    tree->all_intersected_primitives(PrimitiveTraits<PointPrimitive,K>::datum(point), output_it);
  }

  template <class PT>
  void IntersectionOperatorImplementation_d<PT>::all_intersected_entities(const std::vector<Point> & points, uint_set & ids_result) const
  {
    //@remark For a set the start iterator required by the insert_iterator constructor does not really matter.
    std::insert_iterator< uint_set > output_it(ids_result,ids_result.end());
    for (std::vector<Point>::const_iterator p = points.begin(); p != points.end(); ++p)
    {
      tree->all_intersected_primitives(PrimitiveTraits<PointPrimitive,K>::datum(*p), output_it);
    }
  }

  template <class PT> 
  void IntersectionOperatorImplementation_d<PT>::all_intersected_entities(const MeshEntity & entity, std::vector<uint> & ids_result) const
  {
    std::insert_iterator< std::vector<uint> > output_it(ids_result,ids_result.end());
    //Convert entity to corresponding cgal geomtric object according to the mesh
    //entity dimension.
    switch(entity.dim())
    {
      case 0 : tree->all_intersected_primitives(PrimitiveTraits<PointCell,K>::datum(entity), output_it); break;
      case 1 : tree->all_intersected_primitives(PrimitiveTraits<IntervalCell,K>::datum(entity), output_it); break;
      case 2 : tree->all_intersected_primitives(PrimitiveTraits<TriangleCell,K>::datum(entity), output_it); break;
      case 3 : tree->all_intersected_primitives(PrimitiveTraits<TetrahedronCell,K>::datum(entity), output_it); break;
      default:  error("DOLFIN IntersectionOperatorImplementation::all_intersected_entities: \n Mesh CellType is not known."); 
    }
  }

  template <class PT> 
  void IntersectionOperatorImplementation_d<PT>::all_intersected_entities(const std::vector<MeshEntity> & entities, uint_set & ids_result) const
  {
    std::insert_iterator< uint_set > output_it(ids_result,ids_result.end());
    for (std::vector<MeshEntity>::const_iterator entity = entities.begin(); entity != entities.end(); ++entity)
      switch(entity->dim())
      {
        case 0 :  
          tree->all_intersected_primitives(PrimitiveTraits<PointCell,K>::datum(*entity), output_it); break;
        case 1 : 
          tree->all_intersected_primitives(PrimitiveTraits<IntervalCell,K>::datum(*entity), output_it); break;
        case 2 :
          tree->all_intersected_primitives(PrimitiveTraits<TriangleCell,K>::datum(*entity), output_it); break;
        case 3 :
          tree->all_intersected_primitives(PrimitiveTraits<TetrahedronCell,K>::datum(*entity), output_it); break;
        default:  error("DOLFIN IntersectionOperatorImplementation::all_intersected_entities: \n Mesh EntityType is not known."); 
      }
  }

  template <class PT>
  void IntersectionOperatorImplementation_d<PT>::all_intersected_entities(const Mesh & another_mesh, uint_set & ids_result) const
  {
     typedef typename PT::K K;
    //Avoid instantiation of an insert_iterator for each cell.
    std::insert_iterator< uint_set > output_it(ids_result,ids_result.end());
    switch( another_mesh.type().cell_type())
    {
      case CellType::point        : 
        for (CellIterator cell(another_mesh); !cell.end(); ++cell)
          tree->all_intersected_primitives(PrimitiveTraits<PointCell,K>::datum(*cell),output_it); break;
      case CellType::interval     :
        if (dim == 1 || dim == 3)
          dolfin_not_implemented();
        else
          for (CellIterator cell(another_mesh); !cell.end(); ++cell)
            tree->all_intersected_primitives(PrimitiveTraits<IntervalCell,K>::datum(*cell),output_it); break;
      case CellType::triangle     :
        for (CellIterator cell(another_mesh); !cell.end(); ++cell)
          tree->all_intersected_primitives(PrimitiveTraits<TriangleCell,K>::datum(*cell),output_it); break;
      case CellType::tetrahedron  :
          for (CellIterator cell(another_mesh); !cell.end(); ++cell)
          {
            tree->all_intersected_primitives(PrimitiveTraits<TetrahedronCell,K>::datum(*cell),output_it);
          }
          break;
      default:  error("DOLFIN IntersectionOperatorImplementation::all_intersected_entities: \n Mesh CellType is not known."); 
    }
  }

  template <class PT>
  int IntersectionOperatorImplementation_d<PT>::any_intersected_entity(const Point & point) const
  {
    boost::optional<uint> id = tree->any_intersected_primitive(PrimitiveTraits<PointPrimitive,K>::datum(point));
    if (id)
      return *id;
    else 
      return -1;
  }

  template <class PT>
  Point IntersectionOperatorImplementation_d<PT>::closest_point(const Point & point) const
  {
    typedef typename PT::K K;
    if (!point_search_tree_constructed)
      tree->accelerate_distance_queries();
    return Point();
//    return  Point(tree->closest_point(PrimitiveTraits<PointPrimitive,K>::datum(point)));
  }

  template <class PT>
  std::pair<Point,uint> IntersectionOperatorImplementation_d<PT>::closest_point_and_entity_index(const Point & point) const
  {
    typedef typename PT::K K;
//    typedef typename IntersectionOperatorImplementation_d<PT>::Tree Tree;
//    typedef Tree::Point_and_primitive_id Point_and_primitive_id;

//    if (!point_search_tree_constructed)
//      tree->accelerate_distance_queries();
    
//    Point_and_primitive_id pp = tree->closest_point_and_primitive(PrimitiveTraits<PointPrimitive,K>::datum(point));
//    return std::pair<Point,uint>(Point(pp.first),pp.second);
    return std::pair<Point,uint>(Point(),0);
  }

  template <class PT>
  void IntersectionOperatorImplementation_d<PT>::build_tree()
  {
    if (_mesh)
    {
      MeshEntityIterator cell_iter(*_mesh,dim);
      tree.reset(new Tree(cell_iter,cell_iter.end_iterator()));
    }
    point_search_tree_constructed = false;
  }

}

#else

  //Fake interface to allow creation of an IntersectionOperator instance
  //*without* functionality.  IntersectionOperator uses lazy initialization.
  //Throw an exception  if a IntersectionOperatorImplementation instance should
  //be created without CGAL support.
namespace dolfin  {

  class IntersectionOperatorImplementation
  {
  public:
    IntersectionOperatorImplementation() {
      error("DOLFIN has been compiled without CGAL, IntersectionOperatorImplementation is not available.");
    }
    virtual void all_intersected_entities(const Point & point, uint_set & ids_result) const {}  
    virtual void all_intersected_entities(const std::vector<Point> & points, uint_set & ids_result) const {}
    virtual void all_intersected_entities(const MeshEntity & entity, std::vector<uint> & ids_result) const {};
    virtual void all_intersected_entities(const std::vector<MeshEntity> & entities, uint_set & ids_result) const {};
    virtual void all_intersected_entities(const Mesh & another_mesh, uint_set & ids_result) const {}
    virtual int any_intersected_entity(const Point & point) const {return -1; } 
    virtual Point closest_point(const Point & point) const {return Point(); } 
    virtual std::pair<Point,uint> closest_point_and_entity_index(const Point & point) const {return std::pair<Point,uint>(); }

  };
}

#endif 
#endif 
