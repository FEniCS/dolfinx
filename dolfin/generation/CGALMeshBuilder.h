// Copyright (C) 2012 Garth N. Wells
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
// First added:  2012-02-02
// Last changed:

#ifndef __DOLFIN_CGALMESHBUILDER_H
#define __DOLFIN_CGALMESHBUILDER_H

#ifdef HAS_CGAL

#include <vector>

#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/Triangulation_3.h>

#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/Point.h>

namespace dolfin
{

  /// This class provides a function to build a DOLFIN Mesh from a
  /// CGAL triangulation

  class CGALMeshBuilder
  {
  public:

    /// Build DOLFIN Mesh from a CGAL triangulation
    template<typename T>
    static void build(Mesh& mesh, T& triangulation);

    /// Build DOLFIN Mesh from a CGAL mesh (C3t3)
    template<typename T>
    static void build_from_mesh(Mesh& mesh, T& cgal_mesh);

    /// Build DOLFIN Mesh from a CGAL surface mesh (C2t3)
    template<typename T>
    static void build_surface_mesh_c3t3(Mesh& mesh, T& cgal_mesh);

    /// Build DOLFIN Mesh from a CGAL surface mesh (C2t3)
    template<typename T>
    static void build_surface_mesh_c2t3(Mesh& mesh, T& cgal_mesh);

  private:

    // Get number of cells in triangulation (2D)
    template<typename X, typename Y>
    static unsigned int _num_cells(const CGAL::Triangulation_2<X, Y>& t)
    { return t.number_of_faces(); }

    // Get number of cells in Delaunay triangulation (2D)
    template<typename X, typename Y>
    static unsigned int _num_cells(const CGAL::Constrained_Delaunay_triangulation_2<X, Y>& t)
    {
      unsigned int num_cells = 0;
      typename CGAL::Constrained_Delaunay_triangulation_2<X, Y>::Finite_faces_iterator cgal_cell;
      for (cgal_cell = t.finite_faces_begin(); cgal_cell != t.finite_faces_end(); ++cgal_cell)
      {
        if(cgal_cell->is_in_domain())
          ++num_cells;
      }
      return num_cells;
    }

    // Get number of cells in triangulation (3D)
    template<typename X, typename Y>
    static unsigned int _num_cells(const CGAL::Triangulation_3<X, Y>& t)
    { return t.number_of_cells(); }

    // Add cells to mesh (default)
    template<typename T, typename Y>
    static void add_cells(MeshEditor& mesh_editor, const T& t)
    {
      dolfin_error("CGALMeshBuilder.h",
                   "add CGAL cells to DOLFIN Mesh",
                   "Cannot find suitable specialized template funtion");
    }

    // Add cells (from 2D CGAL triangulation)
    template<typename X, typename Y>
    static void add_cells(MeshEditor& mesh_editor,
                          const CGAL::Triangulation_2<X, Y>& t)
    {
      unsigned int cell_index = 0;
      typename CGAL::Triangulation_2<X, Y>::Finite_faces_iterator cgal_cell;
      for (cgal_cell = t.finite_faces_begin(); cgal_cell != t.finite_faces_end(); ++cgal_cell)
      {
        mesh_editor.add_cell(cell_index++, cgal_cell->vertex(0)->info(),
                                           cgal_cell->vertex(1)->info(),
                                           cgal_cell->vertex(2)->info());
      }
    }

    // Add cells (from 2D CGAL constrained Delaunay triangulation)
    template<typename X, typename Y>
    static void add_cells(MeshEditor& mesh_editor,
                          const CGAL::Constrained_Delaunay_triangulation_2<X, Y>& t)
    {
      unsigned int cell_index = 0;
      typename CGAL::Constrained_Delaunay_triangulation_2<X, Y>::Finite_faces_iterator cgal_cell;
      for (cgal_cell = t.finite_faces_begin(); cgal_cell != t.finite_faces_end(); ++cgal_cell)
      {
        // Add cell if it is in the domain
        if(cgal_cell->is_in_domain())
        {
          mesh_editor.add_cell(cell_index++, cgal_cell->vertex(0)->info(),
                                             cgal_cell->vertex(1)->info(),
                                             cgal_cell->vertex(2)->info());
        }
      }
    }

    // Add cells (from 3D CGAL triangulation)
    template<typename X, typename Y>
    static void add_cells(MeshEditor& mesh_editor,
                          const CGAL::Triangulation_3<X, Y>& t)
    {
      unsigned int cell_index = 0;
      typename CGAL::Triangulation_3<X, Y>::Finite_cells_iterator cgal_cell;
      for (cgal_cell = t.finite_cells_begin(); cgal_cell != t.finite_cells_end(); ++cgal_cell)
      {
        mesh_editor.add_cell(cell_index++, cgal_cell->vertex(0)->info(),
                                           cgal_cell->vertex(1)->info(),
                                           cgal_cell->vertex(2)->info(),
                                           cgal_cell->vertex(3)->info());
      }
    }

  };

  //---------------------------------------------------------------------------
  template<typename T>
  void CGALMeshBuilder::build(Mesh& mesh, T& triangulation)
  {
    // Clear mesh
    mesh.clear();

    // Get various dimensions
    const uint gdim = triangulation.finite_vertices_begin()->point().dimension();
    const uint tdim = triangulation.dimension();
    const uint num_vertices = triangulation.number_of_vertices();
    const uint num_cells = _num_cells(triangulation);

    // Create a MeshEditor and open
    dolfin::MeshEditor mesh_editor;
    mesh_editor.open(mesh, tdim, gdim);
    mesh_editor.init_vertices(num_vertices);
    mesh_editor.init_cells(num_cells);

    // Add vertices to mesh
    unsigned int vertex_index = 0;
    typename T::Finite_vertices_iterator cgal_vertex;
    for (cgal_vertex = triangulation.finite_vertices_begin();
            cgal_vertex != triangulation.finite_vertices_end(); ++cgal_vertex)
    {
      // Get vertex coordinates and add vertex to the mesh
      Point p;
      p[0] = cgal_vertex->point()[0];
      p[1] = cgal_vertex->point()[1];
      if (gdim == 3)
        p[2] = cgal_vertex->point()[2];

      // Add mesh vertex
      mesh_editor.add_vertex(vertex_index, vertex_index, p);

      // Attach index to vertex and increment
      cgal_vertex->info() = vertex_index++;
    }

    // Add cells to mesh (calls specialized function because CGAL function
    // named differ in 2D and 3D)
    add_cells(mesh_editor, triangulation);

    // Close mesh editor
    mesh_editor.close();
  }
  //---------------------------------------------------------------------------
  template<typename T>
  void CGALMeshBuilder::build_from_mesh(Mesh& mesh, T& cgal_mesh)
  {
    // Clear mesh
    mesh.clear();

    // CGAL triangulation
    typename T::Triangulation t = cgal_mesh.triangulation();

    // Get various dimensions
    const uint gdim = t.finite_vertices_begin()->point().dimension();
    const uint tdim = t.dimension();
    const uint num_vertices = t.number_of_vertices();
    const uint num_cells = cgal_mesh.number_of_cells();

    // Create a MeshEditor and open
    dolfin::MeshEditor mesh_editor;
    mesh_editor.open(mesh, tdim, gdim);
    mesh_editor.init_vertices(num_vertices);
    mesh_editor.init_cells(num_cells);

    // Add vertices to mesh
    unsigned int vertex_index = 0;
    typename T::Triangulation::Finite_vertices_iterator v;
    for (v = t.finite_vertices_begin(); v != t.finite_vertices_end(); ++v)
    {
      // Get vertex coordinates and add vertex to the mesh
      Point p;
      p[0] = v->point()[0];
      p[1] = v->point()[1];
      p[2] = v->point()[2];

      // Add mesh vertex
      mesh_editor.add_vertex(vertex_index, vertex_index, p);

      // Attach index to vertex and increment
      v->info() = vertex_index++;
    }

    // Sanity check on number of vertices
    dolfin_assert(vertex_index == num_vertices);

    // Iterate over all cell in triangulation
    unsigned int cell_index = 0;
    typename T::Triangulation::Finite_cells_iterator c;
    for (c = t.finite_cells_begin(); c != t.finite_cells_end(); ++c)
    {
      // Add cell if in CGAL mesh, and increment index
      if (cgal_mesh.is_in_complex(c))
      {
        mesh_editor.add_cell(cell_index++, c->vertex(0)->info(),
                                           c->vertex(1)->info(),
                                           c->vertex(2)->info(),
                                           c->vertex(3)->info());
      }
    }

    // Sanity check on number of cells
    dolfin_assert(cell_index == num_cells);

    // Close mesh editor
    mesh_editor.close();
  }
  //---------------------------------------------------------------------------
  template<typename T>
  void CGALMeshBuilder::build_surface_mesh_c3t3(Mesh& mesh, T& cgal_mesh)
  {
    // Clear mesh
    mesh.clear();

    // CGAL triangulation
    typename T::Triangulation t = cgal_mesh.triangulation();

    // Get various dimensions
    const uint gdim = 3;
    const uint tdim = 2;
    const uint num_vertices = t.number_of_vertices();
    const uint num_cells = cgal_mesh.number_of_facets();

    // Create a MeshEditor and open
    dolfin::MeshEditor mesh_editor;
    mesh_editor.open(mesh, tdim, gdim);
    mesh_editor.init_vertices(num_vertices);
    mesh_editor.init_cells(num_cells);

    // Set all vertex indices to -1
    typename T::Facets_in_complex_iterator c;
    for (c = cgal_mesh.facets_in_complex_begin(); c != cgal_mesh.facets_in_complex_end(); ++c)
    {
      c->first->vertex( (c->second + 1)%4 )->info() = -1;
      c->first->vertex( (c->second + 2)%4 )->info() = -1;
      c->first->vertex( (c->second + 3)%4 )->info() = -1;
    }

    unsigned int cell_index = 0;
    unsigned int vertex_index = 0;
    for (c = cgal_mesh.facets_in_complex_begin(); c != cgal_mesh.facets_in_complex_end(); ++c)
    {
      // Add vertex if not already added and increment index
      for (uint i = 1; i < 4; ++i)
      {
        const int v_index = c->first->vertex( (c->second + i)%4 )->info();
        cout << "Testing indices: " << v_index << endl;
        if (v_index < 0)
        {
          c->first->vertex( (c->second + i)%4 )->info() = vertex_index;
          cout << "Testing indices (new): " << v_index << endl;

          // Get vertex coordinates and add vertex to the mesh
          Point p;
          for (uint j = 0; j < 3; ++j)
            p[j] = c->first->vertex((c->second + i) % 4)->point()[j];
          mesh_editor.add_vertex(vertex_index, vertex_index, p);
          ++vertex_index;
        }
      }

      // Get cell vertices and add to Mesh
      std::vector<uint> vertex_indices(3);
      for (uint i = 0; i < 3; ++i)
        vertex_indices[i] = c->first->vertex( (c->second + i + 1)%4 )->info();
      mesh_editor.add_cell(cell_index++, vertex_indices);
    }

    // Close mesh editor
    mesh_editor.close();
  }
  //---------------------------------------------------------------------------
  template<typename T>
  void CGALMeshBuilder::build_surface_mesh_c2t3(Mesh& mesh, T& cgal_mesh)
  {
    // Clear mesh
    mesh.clear();

    // CGAL triangulation
    typename T::Triangulation t = cgal_mesh.triangulation();

    // Get various dimensions
    const uint gdim = 3;
    const uint tdim = 2;
    const uint num_vertices = t.number_of_vertices();
    const uint num_cells = cgal_mesh.number_of_facets();

    cout << "gdim: " << gdim << endl;
    cout << "tdim: " << tdim << endl;
    cout << "num_vert: " << num_vertices << endl;
    cout << "num_cells: " << num_cells << endl;

    // Create a MeshEditor and open
    dolfin::MeshEditor mesh_editor;
    mesh_editor.open(mesh, tdim, gdim);
    mesh_editor.init_vertices(num_vertices);
    mesh_editor.init_cells(num_cells);

    // Add vertices to mesh
    unsigned int vertex_index = 0;
    typename T::Vertex_iterator v;
    for (v = cgal_mesh.vertices_begin(); v != cgal_mesh.vertices_end(); ++v)
    {
      // Get vertex coordinates add vertex to the mesh
      Point p;
      p[0] = v->point()[0];
      p[1] = v->point()[1];
      p[2] = v->point()[2];

      // Add mesh vertex
      mesh_editor.add_vertex(vertex_index, vertex_index, p);

      // Attach index to vertex and increment
      v->info() = vertex_index++;
    }

    unsigned int cell_index = 0;
    typename T::Facet_iterator c;
    for (c = cgal_mesh.facets_begin(); c != cgal_mesh.facets_end(); ++c)
    {
      // Add cell if in CGAL mesh, and increment index
      if (cgal_mesh.is_in_complex(*c))
      {
        std::vector<uint> vertex_indices(3);
        vertex_indices[0] = c->first->vertex( (c->second + 1)%4 )->info();
        vertex_indices[1] = c->first->vertex( (c->second + 2)%4 )->info();
        vertex_indices[2] = c->first->vertex( (c->second + 3)%4 )->info();

        mesh_editor.add_cell(cell_index++, vertex_indices);
      }
    }

    // Close mesh editor
    mesh_editor.close();
  }
  //---------------------------------------------------------------------------

}

#endif
#endif
