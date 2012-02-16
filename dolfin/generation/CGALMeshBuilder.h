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

    /// Build DOLFIN Mesh from a CGAL mesh
    template<typename T>
    static void build_from_mesh(Mesh& mesh, T& cgal_mesh);

  private:

    // Get number of cells in triangulation (2D)
    template<typename X, typename Y>
    static unsigned int _num_cells(const CGAL::Triangulation_2<X, Y>& t)
    { return t.number_of_faces(); }

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
      // Get vertex coordinates add vertex to the mesh
      Point p;
      p[0] = cgal_vertex->point()[0];
      p[1] = cgal_vertex->point()[1];
      if (gdim == 3)
        p[2] = cgal_vertex->point()[2];

      // Add mesh vertex
      mesh_editor.add_vertex(vertex_index, p);

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
      // Get vertex coordinates add vertex to the mesh
      Point p;
      p[0] = v->point()[0];
      p[1] = v->point()[1];
      p[2] = v->point()[2];

      // Add mesh vertex
      mesh_editor.add_vertex(vertex_index, p);

      // Attach index to vertex and increment
      v->info() = vertex_index++;
    }

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

    // Close mesh editor
    mesh_editor.close();
  }
  //---------------------------------------------------------------------------

}

#endif
#endif
