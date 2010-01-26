// =====================================================================================
//
// Copyright (C) 2010-01-16  André Massing
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by André Massing, 2010
//
// First added:  2010-01-16
// Last changed: 2010-01-25
// 
//Author:  André Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
// =====================================================================================

#include "OverlappingMeshes.h"
#include <dolfin/log/dolfin_log.h>

using namespace dolfin;

OverlappingMeshes::OverlappingMeshes(const Mesh & mesh1, const Mesh & mesh2)
{
}

void OverlappingMeshes::compute_overlap_map(const Mesh & mesh1, const Mesh & mesh2)
{
  //@todo Check whether data has already been calculated.

  mesh_data_list.push_back(MeshData(mesh_1)); 
  MeshData & mesh_data mesh_data.last()

  boundary_mesh_data_list.push_back(MeshData(boost::shared_ptr<Mesh>(new BoundaryMesh(mesh_1)))); 
  MeshData & boundary_data_1 = boundary_mesh_data.last();

  boundary_mesh_data_list.push_back(MeshData(boost::shared_ptr<Mesh>(new BoundaryMesh(mesh_2)))); 
  MeshData & boundary_data_2 = boundary_mesh_data.last();


  EntityEntitiesMap facet_1_cell_2_map;
  EntityEntitiesMap facet_2_cell_1_map;
  EntityEntitiesMap cell_cell_map;

  CellIterator cut_cell(mesh_1,0);
  CellIterator cut_facet(boundary_data_2._mesh,0);

  //Step 1: 
  //Intersect boundary of mesh_2 with mesh_1.
  //to get the *partially*
  //intersected cells of mesh_1. 
  //This calculates:
  // a) *partially* intersected cells of mesh1
  // b) the exterior facets of mesh_2 which are (part) of the artificial interface.
  // c) *partially* intersected exterior facets of mesh1 and mesh2.

  for (CellIterator cell(*(boundary_data_2._mesh)); !cell.end(); ++cell)
  {
    //If not empty add cell index and intersecting cell index to the map.
    //@todo does not compile: must be dimension dependent!!!!!!!!1

    //!!!!!!!!!!!!!!!NEEDS TO BE IMPLEMENTED 
    facet_2_cell_1_map[cell.index()] = mesh_1.all_intersected_entitities(*cell);
    if (!facet_2_cell_1_map[cell.index()].empty())
     {
       //Iterate of intersected cell of mesh1, find the overlapping cells of
       //mesh 2 and mark cells in mesh1 as partially overlapped.
       for (EntityListIter cell_iter facet_2_cell_1_map[cell.index()].begin(); cell_iter != facet_2_cell_1_map[cell.index()].end(); ++cell_iter)
       {
	 //!!!!!!!!!!!!!!!NEEDS TO BE IMPLEMENTED 
	 cell_cell_map[*cell_iter] =  mesh_2.all_intersected_entitities(cut_cell[*cell_iter]);
	 mesh_data.intersected_domain(*cell_iter) = 1;
       }
	
       //Compute partially overlapped boundary cells of mesh1 and mesh2.
       //
       //@remark: Clarif whether it is faster to check first if any and then
       //if compute indeces or just try to compute immediately and erase if the cell
       //index container is empty. Rational: We want to avoid a copy operator
       //(linear). A "any intersection" test should have the same complexity as
       //a "compute all intersection" if no intersection occurs. If a
       //intersection occurrs, we want to compute all intersection anyway.
       //1. Version Compute right away and deleting (delete operation is amortized constant) if empty
       //2. Version Check first and compute if intersected.
       //3. Compute and assign if not empty
       //@remark What is faster? Recompute intersecting cells from mesh2 for the
       //exterior facets in mesh1 or map facet index to cell index, and assign
       //their cell set (which might be bigger as the set, which really only
       //intersects the facet).
       
       //!!!!!!!!!!!!!!!NEEDS TO BE IMPLEMENTED 
       EntityList cut_faces(boundary_mesh_data_2._mesh->all_intersected_entitities(*cell));
       if (!cut_face.empty())
       {
	 boundary_data_2.intersected_domain(*face_iter) = 1;

	 //Compute for each cut exterior facet in mesh1 the cutting cells in
	 //mesh2, mark facet as partially overlapped.
	 for (EntityListIter face_iter = cut_faces.begin(); face_iter != cut_faces.end(); ++face_iter)
	 {
	   facet_1_cell_2_map[*face_iter] = mesh_2.all_intersected_entities(cut_facet[*face_iter]);
	   boundary_data_1.intersected_domain(*face_iter) = 1;
	 }
       }
       else
	 boundary_data.intersected_domain(cell.index()) = 2;
     }
     else
       facet_2_cell_1_map.erase(cell.index());
  }

  //Step 2:
  //Determine all cells of Mesh 1, which are fully overlapped. This is done by
  //going through the cells, check if they are not partially overlapped and
  //therefore  must then be fully overlapped if any vertex is intersecting
  //mesh2.
  
  for (CellIterator cell(mesh_1); !cell.end(); ++cell)
    if (mesh_data.intersected_domain(cell.index()) != 1 && mesh_2.any_intersection(Vertex(*cell).point()) != -1)
      mesh_data.intersected_domain(cell_index()) = 2;

  //Step 3:
  //Determine all cells of the boundary of mesh 1, which are fully overlapped.
  //Same method as in Step 2.
  for (CellIterator cell(*(boundary_data_1._mesh)); !cell.end(); ++cell)
    if (boundary_data_1.intersected_domain(cell.index()) != 1 && mesh_2.any_intersection(Vertex(*cell).point()) != -1)
      boundary_data_1.intersected_domain(cell_index()) = 2;
  }
}
