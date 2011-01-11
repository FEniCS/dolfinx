// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2009.
// Modified by Garth N. Wells, 2010.
//
// First added:  2006-11-01
// Last changed: 2011-01-11

#include <dolfin/math/dolfin_math.h>
#include <dolfin/log/dolfin_log.h>
#include "Mesh.h"
#include "MeshTopology.h"
#include "MeshGeometry.h"
#include "MeshConnectivity.h"
#include "MeshEditor.h"
#include "MeshFunction.h"
#include "Vertex.h"
#include "Edge.h"
#include "Cell.h"
#include "BoundaryMesh.h"
#include "LocalMeshRefinement.h"
#include "RivaraRefinement.h"

using namespace dolfin;

struct cmp2
{
   bool operator()(std::vector<dolfin::uint> const a, std::vector<dolfin::uint> const b)
   {
     if (a[0] == b[0])
       return a[1] < b[1];
     return a[0] < b[0];
   }
};

//-----------------------------------------------------------------------------
void LocalMeshRefinement::refineMeshByEdgeBisection(Mesh& refined_mesh,
                                                    const Mesh& mesh,
                                                    const MeshFunction<bool>& cell_marker,
                                                    bool refine_boundary)
{
  info(TRACE, "Refining simplicial mesh by edge bisection.");

  // Copy cell markers
  MeshFunction<bool> old_cell_marker = cell_marker;

  // Get size of old mesh
  const uint num_vertices = mesh.num_vertices();
  const uint num_cells    = mesh.num_cells();

  // Check cell marker
  if (old_cell_marker.size() != num_cells)
    error("Wrong dimension of cell_marker");

  // Generate cell - edge connectivity if not generated
  mesh.init(mesh.topology().dim(), 1);

  // Generate edge - vertex connectivity if not generated
  mesh.init(1, 0);

  // Get cell type
  const CellType& cell_type = mesh.type();

  // Init new vertices and cells
  uint num_new_vertices = 0;
  uint num_new_cells = 0;

  // Create new mesh and open for editing
  MeshEditor editor;
  editor.open(refined_mesh, cell_type.cell_type(),
              mesh.topology().dim(), mesh.geometry().dim());

  // Initialize mappings
  std::vector<int> old2new_cell(mesh.num_cells());
  std::vector<int> old2new_vertex(mesh.num_vertices());

  // Initialise forbidden edges
  MeshFunction<bool> edge_forbidden(mesh, false);

  // If refinement of boundary is forbidden
  if (!refine_boundary)
  {
    BoundaryMesh boundary(mesh);
    for (EdgeIterator e(boundary); !e.end(); ++e)
      edge_forbidden[*e] = true;
  }

  // Initialise forbidden cells
  MeshFunction<bool> cell_forbidden(mesh, mesh.topology().dim(), false);

  // Initialise data for finding longest edge
  uint longest_edge_index = 0;
  double lmax = 0.0;

  // Compute number of vertices and cells
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if (old_cell_marker[*c] && !cell_forbidden[*c])
    {
      // Find longest edge of cell c
      for (EdgeIterator e(*c); !e.end(); ++e)
      {
        if (!edge_forbidden[*e])
        {
          const double l = e->length();
          if (lmax < l)
          {
            lmax = l;
            longest_edge_index = e->index();
          }
        }
      }

      Edge longest_edge(mesh,longest_edge_index);

      // If at least one edge should be bisected
      if (lmax > 0.0)
      {
        // Add new vertex
        num_new_vertices++;

        for (CellIterator cn(longest_edge); !cn.end(); ++cn)
        {
          if (!cell_forbidden[*cn])
          {
            // Count new cells
            num_new_cells++;
            // set markers of all cell neighbors of longest edge to false
            cell_forbidden[*cn] = true;
            // set all the edges of cell neighbors to forbidden
            for (EdgeIterator en(*cn); !en.end(); ++en)
              edge_forbidden[*en] = true;
          }
        }
      }
    }
  }

  // Specify number of vertices and cells
  editor.init_vertices(num_vertices + num_new_vertices);
  editor.init_cells(num_cells + num_new_cells);

  // Add old vertices
  uint current_vertex = 0;
  for (VertexIterator v(mesh); !v.end(); ++v)
    editor.add_vertex(current_vertex++, v->point());

  // Add old unrefined cells
  uint current_cell = 0;
  std::vector<uint> cell_vertices(cell_type.num_entities(0));
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if (!cell_forbidden[*c])
    {
      uint cv = 0;
      for (VertexIterator v(*c); !v.end(); ++v)
        cell_vertices[cv++] = v->index();
      editor.add_cell(current_cell++, cell_vertices);
    }
  }

  // Reset forbidden edges
  edge_forbidden = false;

  // If refinement of boundary is forbidden
  if (!refine_boundary)
  {
    BoundaryMesh boundary(mesh);
    for (EdgeIterator e(boundary); !e.end(); ++e)
      edge_forbidden[*e] = true;
  }

  // Reset forbidden cells
  cell_forbidden = false;

  // Add new vertices and cells.
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if (cell_marker[*c] && !cell_forbidden[*c])
    {
      // Find longest edge of cell c
      lmax = 0.0;
      for (EdgeIterator e(*c); !e.end(); ++e)
      {
        if (edge_forbidden[*e] == false)
        {
          const double l = e->length();
          if (lmax < l)
          {
            lmax = l;
            longest_edge_index = e->index();
          }
        }
      }

      Edge longest_edge(mesh, longest_edge_index);

      // If at least one edge should be bisected
      if (lmax > 0.0)
      {
        // Add new vertex
        editor.add_vertex(current_vertex++, longest_edge.midpoint());

       for (CellIterator cn(longest_edge); !cn.end(); ++cn)
       {
         // Add new cell
         bisect_simplex_edge(*cn, longest_edge, current_vertex, editor, current_cell);

         // Set markers of all cell neighbors of longest edge to false
         old_cell_marker[*cn] = false;

         // Set all edges of cell neighbors to forbidden
         for (EdgeIterator en(*cn); !en.end(); ++en)
           edge_forbidden[*en] = true;
        }
      }
    }
  }

  // Close editor
  editor.close();
}
//-----------------------------------------------------------------------------
void LocalMeshRefinement::refineIterativelyByEdgeBisection(Mesh& refined_mesh,
                                                           const Mesh& mesh,
                                                           const MeshFunction<bool>& cell_marker)
{
  MeshFunction<uint> edges(mesh, 1, 0);

  begin("Iterative mesh refinement algorithm");
  bool more_iterations = true;
  MeshFunction<bool> new_cell_marker;
  while (more_iterations)
  {
    more_iterations = iteration_of_refinement(refined_mesh, cell_marker,
                                              new_cell_marker, edges);
  }
  end();
}
//-----------------------------------------------------------------------------
void LocalMeshRefinement::refineRecursivelyByEdgeBisection(Mesh& refined_mesh,
                                                           const Mesh& mesh,
                                                           const MeshFunction<bool>& cell_marker)
{
  // Transformation maps
  MeshFunction<dolfin::uint> cell_map;
  std::vector<int> facet_map;

  // Create new mesh
  RivaraRefinement::refine(refined_mesh, mesh, cell_marker, cell_map, facet_map);
  transform_data(refined_mesh, mesh, cell_map, facet_map);
}
//-----------------------------------------------------------------------------
void LocalMeshRefinement::bisect_simplex_edge(const Cell& cell, const Edge& edge,
                                              uint new_vertex,
                                              MeshEditor& editor,
                                              uint& current_cell)
{
  // Init cell vertices
  std::vector<uint> cell1_vertices(cell.num_entities(0));
  std::vector<uint> cell2_vertices(cell.num_entities(0));

  // Get edge vertices
  const uint* edge_vert = edge.entities(0);

  uint vc1 = 0;
  uint vc2 = 0;

  for (VertexIterator v(cell); !v.end(); ++v)
  {
    if ((v->index() != edge_vert[0]) && (v->index() != edge_vert[1]))
    {
      cell1_vertices[vc1++] = v->index();
      cell2_vertices[vc2++] = v->index();
    }
  }

  cell1_vertices[vc1++] = new_vertex - 1;
  cell2_vertices[vc2++] = new_vertex - 1;

  cell1_vertices[vc1++] = edge_vert[0];
  cell2_vertices[vc2++] = edge_vert[1];

  editor.add_cell(current_cell++, cell1_vertices);
  editor.add_cell(current_cell++, cell2_vertices);
}
//-----------------------------------------------------------------------------
bool LocalMeshRefinement::iteration_of_refinement(Mesh& mesh,
                                         const MeshFunction<bool>& cell_marker,
                                         MeshFunction<bool>& new_cell_marker,
                                         MeshFunction<uint>& bisected_edges)
{
  // Map used for edge function transformation
  std::map<std::vector<uint>, uint, cmp2> edge_map;
  std::map<std::vector<uint>, uint, cmp2>::iterator edge_map_it;

  const uint num_vertices = mesh.size(0);
  const uint num_cells = mesh.size(mesh.topology().dim());
  const CellType& cell_type = mesh.type();

  Mesh refined_mesh;
  MeshEditor editor;
  editor.open(refined_mesh, cell_type.cell_type(),
              mesh.topology().dim(), mesh.geometry().dim());

  // Generate cell - edge connectivity if not generated
  mesh.init(mesh.topology().dim(), 1);

  // Generate edge - vertex connectivity if not generated
  mesh.init(1, 0);

  uint num_new_vertices = 0;
  uint num_new_cells = 0;
  uint longest_edge_index = 0;
  double lmax, l;

  // Temporary edge marker function
  MeshFunction<bool> found_edge(mesh, 1, false);

  // Calculate number of new vertices
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if (cell_marker[*c])
    {
      // Find longest edge of cell c
      lmax = 0.0;
      for(EdgeIterator e(*c); !e.end(); ++e)
      {
        l = e->length();
        if (lmax < l)
        {
          lmax = l;
          longest_edge_index = e->index();
        }
      }
      Edge longest_edge(mesh,longest_edge_index);
      uint middle_vertex = bisected_edges[longest_edge_index];
      if (middle_vertex == 0 && !found_edge[longest_edge_index])
      {
        found_edge[longest_edge_index] = true;
        num_new_vertices++;
      }
      num_new_cells++;
    }
  }

  cout << "Number of cells in old mesh: " << num_cells << "; to add: " << num_new_cells << endl;
  cout << "Number of vertices in old mesh: " << num_vertices << "; to add: " << num_new_vertices << endl;

  editor.init_vertices(num_vertices + num_new_vertices);
  editor.init_cells(num_cells + num_new_cells);

  // Rewrite old vertices
  uint current_vertex = 0;
  for (VertexIterator v(mesh); !v.end(); ++v)
    editor.add_vertex(current_vertex++, v->point());

  // Rewrite old cells
  uint current_cell = 0;
  std::vector<uint> cell_vertices(cell_type.num_entities(0));
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if (!cell_marker[*c])
    {
      uint cv = 0;
      for (VertexIterator v(*c); !v.end(); ++v)
        cell_vertices[cv++] = v->index();
      editor.add_cell(current_cell++, cell_vertices);
    }
  }

  // Main loop
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if (cell_marker[*c])
    {

      // Find longest edge of cell c
      lmax = 0.0;
      for(EdgeIterator e(*c); !e.end(); ++e)
      {
        l = e->length();
        if (lmax < l)
        {
          lmax = l;
          longest_edge_index = e->index();
        }
      }

      Edge longest_edge(mesh,longest_edge_index);

      uint middle_vertex = bisected_edges[longest_edge_index];
      if (middle_vertex == 0)
      {
        // Add new vertex
        editor.add_vertex(current_vertex++, longest_edge.midpoint());

        middle_vertex = current_vertex-1;

        // Insert new vertex into the bisected edges mapping
        std::vector<uint> ev(2);
        ev[0] = longest_edge.entities(0)[0];
        ev[1] = longest_edge.entities(0)[1];
        edge_map[ev] = middle_vertex;

        bisected_edges[longest_edge_index] = middle_vertex;
      }

      // Add two new cells
      bisect_simplex_edge(*c, longest_edge, middle_vertex+1, editor, current_cell);

    }
  }
  editor.close();

  // Rewrite old bisected edges
  for (EdgeIterator e(mesh); !e.end(); ++e)
  {
    const uint eix = bisected_edges[*e];
    if (eix)
    {
      std::vector<uint> ev(2);
      ev[0] = e->entities(0)[0];
      ev[1] = e->entities(0)[1];
      edge_map[ev] = eix;
    }
  }

  // Initialize new cell markers
  new_cell_marker.init(refined_mesh, refined_mesh.topology().dim());
  new_cell_marker = false;

  // Initialize new bisected edges
  bisected_edges.init(refined_mesh, 1);
  bisected_edges = 0;

  Progress p("Calculate cell markers and bisected edges", refined_mesh.num_edges());
  bool next_iteration = false;
  for (EdgeIterator e(refined_mesh); !e.end(); ++e)
  {
    std::vector<uint> ev(2);
    ev[0]= e->entities(0)[0];
    ev[1]= e->entities(0)[1];
    if ((edge_map_it = edge_map.find(ev)) == edge_map.end())
      bisected_edges[*e] = 0;
    else
    {
      bisected_edges[*e] = edge_map_it->second;
      for(CellIterator c(*e); !c.end(); ++c)
      {
        new_cell_marker[*c] = true;
        next_iteration = true;
      }
    }
    p++;
  }

  // Overwrite old mesh
  mesh = refined_mesh;

  return next_iteration;
}
//-----------------------------------------------------------------------------
void LocalMeshRefinement::transform_data(Mesh& newmesh, const Mesh& oldmesh,
                                         const MeshFunction<uint>& cell_map,
                                         const std::vector<int>& facet_map)
{
  newmesh.data().clear();

  // Rewrite materials
  if (oldmesh.data().mesh_function("material indicators"))
  {
    MeshFunction<dolfin::uint>* mat;
    mat = newmesh.data().create_mesh_function("material indicators", newmesh.type().dim());
    for(dolfin::uint i=0; i< newmesh.num_cells(); i++)
      (*mat)[i] = (*oldmesh.data().mesh_function("material indicators"))[cell_map[i]];
    info(TRACE, "MeshData MeshFunction \"material indicators\" transformed.");
  }

  // Rewrite boundary indicators
  if (oldmesh.data().array("boundary facet cells")
      && oldmesh.data().array("boundary facet numbers")
      && oldmesh.data().array("boundary indicators"))
  {

    dolfin::uint num_ent = oldmesh.type().num_entities(0);
    std::vector<dolfin::uint>* bfc;
    std::vector<dolfin::uint>* bfn;
    std::vector<dolfin::uint>* bi ;
    bfc = oldmesh.data().array("boundary facet cells");
    bfn = oldmesh.data().array("boundary facet numbers");
    bi  = oldmesh.data().array("boundary indicators");
    dolfin::uint bi_table_size = oldmesh.num_cells()*num_ent;
    std::vector<int> bi_table;
    bi_table.resize(bi_table_size);
    for(dolfin::uint i=0; i< bi_table_size; i++)
      bi_table[i] = -1;
    for(dolfin::uint i=0; i< bi->size(); i++)
      bi_table[ (*bfc)[i]*num_ent+(*bfn)[i] ] = (*bi)[i];

    // Empty loop to count elements
    dolfin::uint bi_size = 0;
    for(dolfin::uint c=0; c< newmesh.num_cells(); c++)
    {
      for(dolfin::uint f=0; f< num_ent; f++)
      {
        if (facet_map[ c*num_ent+f ] != -1)
        {
          dolfin::uint table_map = cell_map[c]*num_ent + facet_map[c*num_ent+f];
          if (bi_table[ table_map ] != -1)
            bi_size++;
        }
      }
    }

    // Create new MeshData std::vectors for boundary indicators
    std::vector<dolfin::uint>* bfc_new;
    std::vector<dolfin::uint>* bfn_new;
    std::vector<dolfin::uint>* bi_new ;
    bfc_new = newmesh.data().create_array("boundary facet cells", bi_size);
    bfn_new = newmesh.data().create_array("boundary facet numbers", bi_size);
    bi_new  = newmesh.data().create_array("boundary indicators", bi_size);

    // Main transformation loop
    dolfin::uint number_bi = 0;
    for(dolfin::uint c=0; c< newmesh.num_cells(); c++)
    {
      for(dolfin::uint f=0; f< num_ent; f++)
      {
        if (facet_map[ c*num_ent+f ] != -1)
        {
          dolfin::uint table_map = cell_map[c]*num_ent + facet_map[c*num_ent+f];
          if (bi_table[ table_map ] != -1)
          {
            (*bfc_new)[number_bi] = c;
            (*bfn_new)[number_bi] = f;
            (*bi_new)[number_bi] = bi_table[table_map];
            number_bi++;
          }
        }
      }
    }
    info(TRACE, "MeshData \"boundary indicators\" transformed.");
  }
}
//-----------------------------------------------------------------------------
