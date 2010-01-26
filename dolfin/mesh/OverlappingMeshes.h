// =====================================================================================
//
// Copyright (C) 2010-01-15  André Massing
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by André Massing, 2010
//
// First added:  2010-01-15
// Last changed: 2010-01-25
// 
//Author:  André Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
// =====================================================================================


#ifndef  __OVERLAPPINGMESHES_H
#define  __OVERLAPPINGMESHES_H

#include <vector>
#include <map>
#include <pair>

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

#include <dolfin/common/Array.h>
#include <dolfin/common/types.h>
#include "Mesh.h"

namespace dolfin
{
  class Mesh;

  typedef std::vector<uint> EntityList;
  typedef std::vector<uint>::const_iterator EntityListIter;
  typedef std::map<uint, EntityList> EntityEntitiesMap;
  typedef std::map<const Mesh *,EntityEntitiesMap> MeshCutEntitiesMap;

  ///This class present a collection of overlapping meshes and provides
  ///functionality to compute cell-cell, cell-facet overlaps.
  class OverlappingMeshes {
    
    class MeshData {
    public:
      MeshData(boost::shared_ptr<Mesh> mesh) : _mesh(mesh), intersected_domain(*mesh) {}

//    private:
      MeshCutEntitiesMap entity_entities_map;
      MeshFunction<uint> intersected_domain;
      boost::shared_ptr<Mesh> _mesh;

    };

  public:
  

    ///Constructor takes a list/vector of meshes. The order of meshes defines
    ///also the "overlapp" priority, i.e. if 2 meshes overlap, the one who
    //appears later in the list actually covers the later one.
//    OverlappingMeshes(Array<Mesh*> meshes);
    OverlappingMeshes(const Mesh & mesh1, const Mesh & mesh2);

    ///Computes the overlap mapping. Mesh2  overlaps mesh1. Computes (for
    ///efficient reasons) in addition the boundary overlaps and the artificial
    ///interface.
    void compute_overlap_map(const Mesh & mesh1, const Mesh & mesh2);


  private:
    std::vector<MeshData> mesh_data_list;
    std::vector<MeshData> boundary_mesh_data_list;
    Array<const Mesh *> _meshes;

  };


} //end namespace dolfin    

#endif   // ----- #ifndef __OVERLAPPINGMESHES_H  -----
