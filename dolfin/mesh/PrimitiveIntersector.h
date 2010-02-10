// =====================================================================================
//
// Copyright (C) 2010-02-09  André Massing
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by André Massing, 2010
//
// First added:  2010-02-09
// Last changed: 2010-02-10
// 
//Author:  André Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
// =====================================================================================

#ifndef  __PRIMITIVEINTERSECTOR_H
#define  __PRIMITIVEINTERSECTOR_H

namespace dolfin
{
  class MeshEntity;

  /// This class implements an intersection detection, detecting whether two given (arbitrary) meshentities intersect.
  class PrimitiveIntersector
  {
    public:
      static bool do_intersect(const MeshEntity & entity_1, const MeshEntity & entity_2);
      static bool do_intersect_exact(const MeshEntity & entity_1, const MeshEntity & entity_2);
    private:
      
      //Helper classes to deal with all combination in a n 2*N and not N*N way. 
      //Just declaration, definition and instantation takes place in the corresponding cpp file, where
      //this helper function are actually needed.
      //
      template<typename Kernel> 
      static bool do_intersect_with_kernel(const MeshEntity & entity_1, const MeshEntity & entity_2);

      template<typename T> 
      static bool _do_intersect(const T & entity_1, const MeshEntity & entity_2);

  };
}


#endif   // ----- #ifndef __PRIMITIVEINTERSECTOR_H  -----

