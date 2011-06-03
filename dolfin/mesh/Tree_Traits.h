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
