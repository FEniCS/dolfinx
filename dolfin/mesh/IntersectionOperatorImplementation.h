// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2009-11-29

#ifndef __INTERSECTIONOPERATORIMPLEMENTATION_H
#define __INTERSECTIONOPERATORIMPLEMENTATION_H

#ifdef HAS_CGAL

#include <vector>

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bimap.hpp>
#include <boost/optional.hpp>

#include <dolfin/common/types.h>

#include "added_intersection_3.h" //additional intersection functionality, *Must* include before the AABB_tree!

#include <CGAL/AABB_tree.h> // *Must* be inserted before kernel!
#include <CGAL/AABB_traits.h>

#include <CGAL/Simple_cartesian.h> 
#include "Triangle_3_Tetrahedron_3_do_intersect_SCK.h" //template specialization for Simple_cartesian kernel

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Bbox_3.h>
#include <CGAL/Point_3.h>


#include "Primitive_Traits.h"
#include "MeshPrimitive.h"


typedef CGAL::Simple_cartesian<double> SCK;
typedef CGAL::Exact_predicates_inexact_constructions_kernel EPICK;

namespace dolfin
{

  class Point;
  class Mesh;

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
    virtual void all_intersected_entities(const Mesh & another_mesh, uint_set & ids_result) const = 0;
    virtual int any_intersected_entity(const Point & point) const = 0; 

  };

  template <class PrimitiveTrait>
  class IntersectionOperatorImplementation_d : public IntersectionOperatorImplementation
  {
    typedef PrimitiveTrait PT;
    typedef typename PT::K K;
    typedef MeshPrimitive<PrimitiveTrait> CellPrimitive;

//    typedef Tree_Traits<K,CellPrimitive> AABB_PrimitiveTraits;
    typedef CGAL::AABB_traits<K,CellPrimitive> AABB_PrimitiveTraits;
    typedef CGAL::AABB_tree<AABB_PrimitiveTraits> Tree;

  public:
    ///Constructor. 
    IntersectionOperatorImplementation_d(boost::shared_ptr<const Mesh> _mesh) : _mesh(_mesh) 
    {
      build_tree();
    }

    virtual void all_intersected_entities(const Point & point, uint_set & ids_result) const; 
    virtual void all_intersected_entities(const std::vector<Point> & points, uint_set & ids_result) const;
    virtual void all_intersected_entities(const Mesh & another_mesh, uint_set & ids_result) const;

    virtual  int any_intersected_entity(const Point & point) const;

    ///Topological dimension of the mesh.
    static const uint dim = PrimitiveTrait::dim;

  private:

    void build_tree();
    boost::shared_ptr<const Mesh> _mesh;
    boost::scoped_ptr<Tree> tree;

  };

  template <class PT>
  void IntersectionOperatorImplementation_d<PT>::all_intersected_entities(const Point & point, uint_set & ids_result) const
  {
    //@remark For a set the start iterator required by the insert_iterator constructor does not really matter.
    std::insert_iterator< uint_set > output_it(ids_result,ids_result.end());
    tree->all_intersected_primitives(Primitive_Traits<PointPrimitive,K>::datum(point), output_it);
  }

  template <class PT>
  void IntersectionOperatorImplementation_d<PT>::all_intersected_entities(const std::vector<Point> & points, uint_set & ids_result) const
  {
    //@remark For a set the start iterator required by the insert_iterator constructor does not really matter.
    std::insert_iterator< uint_set > output_it(ids_result,ids_result.end());
    for (std::vector<Point>::const_iterator p = points.begin(); p != points.end(); ++p)
    {
      tree->all_intersected_primitives(Primitive_Traits<PointPrimitive,K>::datum(*p), output_it);
    }
  }

//  template< template <class PT>
//  void IntersectionOperatorImplementation_d<PT>::all_intersected_entities(const Point & point, uint_set & ids_result) const

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
	  tree->all_intersected_primitives(Primitive_Traits<PointCell,K>::datum(*cell),output_it); break;
      case CellType::interval     :
	if (dim == 1 || dim == 3)
	  dolfin_not_implemented();
	else
	  for (CellIterator cell(another_mesh); !cell.end(); ++cell)
	    tree->all_intersected_primitives(Primitive_Traits<IntervalCell,K>::datum(*cell),output_it); break;
      case CellType::triangle     :
	for (CellIterator cell(another_mesh); !cell.end(); ++cell)
	  tree->all_intersected_primitives(Primitive_Traits<TriangleCell,K>::datum(*cell),output_it); break;
      case CellType::tetrahedron  :
	  for (CellIterator cell(another_mesh); !cell.end(); ++cell)
	  {
	    tree->all_intersected_primitives(Primitive_Traits<TetrahedronCell,K>::datum(*cell),output_it);
	  }
	  break;
      default:  error("DOLFIN IntersectionOperatorImplementation::all_intersected_entities: \n Mesh CellType is not known."); 
    }
  }

  template <class PT>
  int IntersectionOperatorImplementation_d<PT>::any_intersected_entity(const Point & point) const
  {
    boost::optional<uint> id = tree->any_intersected_primitive(Primitive_Traits<PointPrimitive,K>::datum(point));
    if (id)
      return *id;
    else 
      return -1;
  }

  template <class PT>
  void IntersectionOperatorImplementation_d<PT>::build_tree()
  {
    if (_mesh)
    {
      MeshEntityIterator cell_iter(*_mesh,dim);
      tree.reset(new Tree(cell_iter,cell_iter.end_iterator()));
    }
  }

}

#endif 
#endif 
