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
// Last changed: 2012-06-02

#ifndef __VTKPLOTTER_H
#define __VTKPLOTTER_H

#ifdef HAS_VTK

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPointSet.h>
#include <vtkPolyDataAlgorithm.h>
#include <vtkWarpScalar.h>
#include <vtkWarpVector.h>
#include <vtkGlyph3D.h>
#include <vtkGeometryFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
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
      p.add("interactive", true);
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

    // Set up help text and start interaction loop 
    void interactive();

    // Return unique ID of the object to plot
    uint id() const { return _id; }

  private:

    // Setup all pipeline objects and connect them. Called from all 
    // constructors
    void init_pipeline();

    // Construct VTK grid from DOLFIN mesh
    void construct_vtk_grid();
    
    // Process mesh (including filter setup)
    void process_mesh();

    // Process scalar valued generic function (function or expression),
    // including filter setup
    void process_scalar_function();
    
    // Process vector valued generic function (function or expression),
    // including filter setup
    void process_vector_function();

    // Return the hover-over help text
    std::string get_helptext();

    // The mesh to visualize
    boost::shared_ptr<const Mesh> _mesh;

    // The (optional) function values to visualize
    boost::shared_ptr<const GenericFunction> _function;

    // The VTK grid constructed from the DOLFIN mesh
    vtkSmartPointer<vtkUnstructuredGrid> _grid;

    // The scalar warp filter
    vtkSmartPointer<vtkWarpScalar> _warpscalar;

    // The vector warp filter
    vtkSmartPointer<vtkWarpVector> _warpvector;

    // The glyph filter 
    vtkSmartPointer<vtkGlyph3D> _glyphs;

    // The geometry filter
    vtkSmartPointer<vtkGeometryFilter> _geometryFilter;

    // The poly data mapper
    vtkSmartPointer<vtkPolyDataMapper> _mapper;

    // The lookup table
    vtkSmartPointer<vtkLookupTable> _lut;

    // The main actor
    vtkSmartPointer<vtkActor> _actor;

    // The renderer
    vtkSmartPointer<vtkRenderer> _renderer;

    // The render window
    vtkSmartPointer<vtkRenderWindow> _renderWindow;

    // The render window interactor for interactive sessions
    vtkSmartPointer<vtkRenderWindowInteractor> _interactor;

    // The scalar bar that gives the viewer the mapping from color to 
    // scalar value
    vtkSmartPointer<vtkScalarBarActor> _scalarBar;

    // The unique ID (inherited from Variable) for the object to plot
    uint _id;

  };

}

#endif

#endif
