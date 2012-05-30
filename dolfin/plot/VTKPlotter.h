// Copyright (C) 2012 Fredrik Valdmanis
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
// Modified by Benjamin Kehlet, 2012
//
// First added:  2012-05-23
// Last changed: 2012-05-30

#ifndef __VTKPLOTTER_H
#define __VTKPLOTTER_H

#ifdef HAS_VTK

#include <vtkSmartPointer.h>
#include <vtkPointSet.h>
#include <vtkPolyDataAlgorithm.h>
#include <vtkActor.h>
#include <vtkUnstructuredGrid.h>
#include <vtkScalarBarActor.h>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/Expression.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  /// This is a class for visualizing of DOLFIN meshes, functions, 
  /// expressions and mesh functions. 
  /// The plotter has several parameters that the user can change to control
  /// the appearance and behavior of the plot. 
  ///
  /// TODO: Add documentation for all parameters?

  class VTKPlotter : public Variable
  {
  public:

    /// Create plotter for a mesh
    explicit VTKPlotter(const Mesh& mesh);

    /// Create plotter for a function
    explicit VTKPlotter(const Function& function);

    /// Create plotter for an expression
    explicit VTKPlotter(const Expression& expression, const Mesh& mesh);

    /// Create plotter for an integer valued mesh function
    explicit VTKPlotter(const MeshFunction<uint>& mesh_function);

    /// Create plotter for a double valued mesh function
    explicit VTKPlotter(const MeshFunction<double>& mesh_function);

    /// Create plotter for a boolean valued mesh function
    explicit VTKPlotter(const MeshFunction<bool>& mesh_function);

    /// Plot the object
    void plot();

    ~VTKPlotter();

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("vtk_plotter");
      p.add("title", "Plot");
      p.add("wireframe", false);
      p.add("scalarbar", true);
      p.add("vector_mode", "glyphs");
      p.add("warp_scalefactor", 1.0);
      p.add("glyph_scalefactor", 0.8);
      return p;
    }

    // TODO: Add separate default parameters for each type?
    // TODO: Set title in default_parameters to "Mesh", "Expression" etc
    // as in plot.h?

    /// Default parameter values for mesh plotting
    static Parameters default_mesh_parameters()
    {
      Parameters p = default_parameters();
      p["wireframe"] = true;
      p["scalarbar"] = false;
      return p;
    }

  private:

    // Construct VTK grid from DOLFIN mesh
    void construct_vtk_grid();
    
    // Plot scalar valued generic function (function or expression)
    void plot_scalar_function();
    
    // Plot vector valued generic function (function or expression)
    void plot_vector_function();

    // Plot vector valued function with warp (displacement) visualization
    void plot_warp();

    // Plot vector valued function using visualization with glyphs (vectors)
    void plot_glyphs();

    // Filter a VTK point set through a geometryfilter and pass it to the map
    // function
    void filter_and_map(vtkSmartPointer<vtkPointSet> point_set);
    
    // Map a VTK poly data into geometric primitives, attach it to a VTK
    // actor and pass the actor to the render function
    void map(vtkSmartPointer<vtkPolyDataAlgorithm> polyData);
    
    // Render the given VTK actor
    void render(vtkSmartPointer<vtkActor> actor);

    // The mesh to visualizae
    boost::shared_ptr<const Mesh> _mesh;

    // The (optional) function values to visualize
    boost::shared_ptr<const GenericFunction> _function;

    // The VTK grid constructed from the DOLFIN mesh
    vtkSmartPointer<vtkUnstructuredGrid> _grid;

    // The scalar bar that gives the viewer the mapping from color to 
    // scalar value
    vtkSmartPointer<vtkScalarBarActor> _scalarbar;

  };

}

#endif

#endif
