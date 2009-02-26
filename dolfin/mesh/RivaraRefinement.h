// Copyright (C) 2008 Johan Jansson
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Bartosz Sawicki, 2009.

#ifndef __RIVARAREFINEMENT_H
#define __RIVARAREFINEMENT_H

#include <list>
#include <vector>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>

namespace dolfin
{

  class RivaraRefinement
  {

  public:
    
    /// Refine simplicial mesh locally by recursive edge bisection 
    static void refine(Mesh& mesh, 
		       MeshFunction<bool>& cell_marker,
		       MeshFunction<uint>& cell_map,
		       std::vector<int>& facet_map);

  private:

    class DCell;
  
    /// Vertex with list of connected cells
    class DVertex
    {
    public:
      DVertex();
      int id;
      std::list<DCell*> cells;
      Point p;
    };
    
    // Cell with parent_id, deletion marker and facets markets
    class DCell
    {
    public:
      DCell();
      int id;
      int parent_id;
      std::vector<DVertex *> vertices;
      bool deleted;
      std::vector<int> facets; 
    };
    
    // Dynamic mesh for recursive Rivara refinement
    class DMesh
    {
    public:
      DMesh();
      ~DMesh();
      void addVertex(DVertex* v);
      void addCell(DCell* c, std::vector<DVertex*> vs, int parent_id);
      void removeCell(DCell* c);
      void importMesh(Mesh& mesh);
      void exportMesh(Mesh& mesh, std::vector<int>& new2old_cell, std::vector<int>& new2old_facet);
      void number();
      void bisect(DCell* dcell, DVertex* hangv, DVertex* hv0, DVertex* hv1);
      void bisectMarked(std::vector<bool> marked_ids);
                        DCell* opposite(DCell* dcell, DVertex* v1, DVertex* v2);
      void propagateFacets(DCell* dcell, DCell* c0, DCell* c1, uint ii, uint jj);

      std::list<DVertex *> vertices;
      std::list<DCell *> cells;
      CellType* cell_type;
      uint dim;
    };


  };




}

#endif
