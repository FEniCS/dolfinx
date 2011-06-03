// Copyright (C) 2008 Johan Jansson
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
// Modified by Bartosz Sawicki, 2009.
// Modified by Garth N. Wells, 2010.
// Modified by Anders Logg, 2010.
//
// First added:  2008
// Last changed: 2011-01-11

#ifndef __RIVARAREFINEMENT_H
#define __RIVARAREFINEMENT_H

#include <list>
#include <vector>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>

namespace dolfin
{
  // Forward declarations
  //class CellType;
  //class Mesh

  // FIXME: This implementation needs quite a bit of cleanup, in particular
  // FIXME: the DMesh class. Should be possible to use plain STL data structures
  // FIXME: in place of the DMesh class.

  class RivaraRefinement
  {

  public:

    /// Refine simplicial mesh locally by recursive edge bisection
    static void refine(Mesh& refined_mesh,
                       const Mesh& mesh,
                       const MeshFunction<bool>& cell_marker,
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

      void add_vertex(DVertex* v);
      void add_cell(DCell* c, std::vector<DVertex*> vs, int parent_id);
      void remove_cell(DCell* c);
      void import_mesh(const Mesh& mesh);
      void export_mesh(Mesh& mesh, std::vector<int>& new2old_cell, std::vector<int>& new2old_facet);
      void number();
      void bisect(DCell* dcell, DVertex* hangv, DVertex* hv0, DVertex* hv1);
      void bisect_marked(std::vector<bool> marked_ids);
      DCell* opposite(DCell* dcell, DVertex* v1, DVertex* v2);
      void propagate_facets(DCell* dcell, DCell* c0, DCell* c1, uint ii, uint jj, DVertex* mv);

      std::list<DVertex*> vertices;
      std::list<DCell*> cells;
      const CellType* cell_type;
      uint tdim;
      uint gdim;

    };

  };

}

#endif
