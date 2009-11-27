// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-10-12
// Last changed: 2009-11-10

#ifndef  __TREE_TRAITS_H
#define  __TREE_TRAITS_H

#include <CGAL/AABB_traits.h>

namespace dolfin 
{

  template<typename GeomTraits, typename AABB_primitive>
  class Tree_Traits:  public CGAL::AABB_traits<GeomTraits, AABB_primitive>
  {

  public:
    typedef CGAL::AABB_traits<GeomTraits, AABB_primitive> AT;
    typedef typename CGAL::Bbox_3 Bounding_box;
    typedef AABB_primitive Primitive;


    //Redefine  this class in order to overwrite static compute_bbox methods
    //and to use our own (calling directly the bbox of primitive, which is not
    //required by th CGAL primitive concept.
    class Compute_bbox {
    public:
      template<typename ConstPrimitiveIterator>
      typename AT::Bounding_box operator()(ConstPrimitiveIterator first,
                                           ConstPrimitiveIterator beyond) const
      {
        typename AT::Bounding_box bbox = compute_bbox(*first);
        for(++first; first != beyond; ++first)
        {
          bbox = bbox + compute_bbox(*first);
        }
        return bbox;
      }
    };

    Compute_bbox compute_bbox_object() {return Compute_bbox();}

  private:
    /**
     * @brief Computes bounding box of one primitive
     * @param pr the primitive
     * @return the bounding box of the primitive \c pr
     */
    static Bounding_box compute_bbox(const Primitive& pr)
    {
      return pr.bbox();
    }
  };

}

#endif   // ----- #ifndef __TREE_TRAITS_H  -----
