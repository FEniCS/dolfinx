// Copyright (C) 2012 Joachim B Haga
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
// First added:  2012-09-10
// Last changed: 2012-09-14

#ifndef __VTK_WINDOW_OUTPUT_STAGE_H
#define __VTK_WINDOW_OUTPUT_STAGE_H

#include <vtkSmartPointer.h>

// Forward declarations
class QVTKWidget;
class vtkActor;
class vtkAlgorithm;
class vtkAlgorithmOutput;
class vtkAxesActor;
class vtkBalloonRepresentation;
class vtkBalloonWidget;
class vtkCamera;
class vtkDepthSortPolyData;
class vtkLookupTable;
class vtkPolyDataMapper;
class vtkRenderWindow;
class vtkRenderWindowInteractor;
class vtkRenderer;
class vtkScalarBarActor;
class vtkTextActor;
class vtkProp;

namespace dolfin
{

  // Forward declarations
  class GenericVTKPlottable;
  class Parameters;
  class VTKPlotter;

  /// This class enables visualization of various DOLFIN entities.
  class VTKWindowOutputStage
  {

  public:

    /// If a widget is supplied, this widget will be used for drawing,
    /// instead of a new top-level widget. Ownership is transferred.
    VTKWindowOutputStage();

    /// Destructor
    ~VTKWindowOutputStage();

    /// Initialise the pipeline
    void init(VTKPlotter* parent, const Parameters& parameters);

    /// Get the vtkRenderWindowInteractor for the window
    vtkRenderWindowInteractor* get_interactor();

    /// Get the vtkRenderer for the scene
    vtkSmartPointer<vtkRenderer> get_renderer();

    /// Scale points and lines by the given factor
    void scale_points_lines(double factor);

    /// Set the help text, and (re)create the popup widget
    void set_helptext(std::string text);

    /// Change the window title
    void set_window_title(std::string title);

    /// Retrieve the window title
    std::string get_window_title();

    /// Start interaction, and optionally enter the event loop.
    void start_interaction(bool enter_eventloop=true);

    /// Exit the event loop
    void stop_interaction();

    /// Write the current frame to raster file
    void write_png(std::string filename);

    /// Write the current frame to vector file
    void write_pdf(std::string filename);

    /// Retrieve the camera
    vtkCamera* get_camera();

    /// Reset the camera to cover the whole scene
    void reset_camera();

    /// Reset camera clipping ranges, if the scene has changed
    void reset_camera_clipping_range();

    /// Set the scalar range for colorbar
    void set_scalar_range(double *range);

    /// Cycle between surface--wireframe--points representation
    void cycle_representation(int new_rep=0);

    /// Toggle the bounding box around the main actor
    void toggle_boundingbox();

    /// Toggle the help text box with the given text
    void toggle_helptext(std::string text);

    /// Re-render the current frame
    void render();

    /// Get the size of the plot window
    void get_window_size(int& width, int& height);

    /// Get the size of the screen
    void get_screen_size(int& width, int& height);

    /// Place the plot window at the given coordinates
    void place_window(int x, int y);

    /// Add a prop to the scene. If it is already in the scene,
    /// it will not be re-added.
    bool add_viewprop(vtkSmartPointer<vtkProp> prop);

    /// Set the input for the output stage.
    void set_input(vtkSmartPointer<vtkAlgorithmOutput> output);

    /// Used by plottables to indicate whether the scene should be
    /// treated as translucent (which requires depth sorting, etc.)
    void set_translucent(bool onoff, std::size_t topo_dim=3,
                         std::size_t geom_dim=3);

  protected:

    // The depth sorting filter
    vtkSmartPointer<vtkDepthSortPolyData> _depthSort;

    // The poly data mapper
    vtkSmartPointer<vtkPolyDataMapper> _mapper;

    // The input port (either the mapper or depth sorter)
    vtkSmartPointer<vtkAlgorithm> _input;

    // The lookup table
    vtkSmartPointer<vtkLookupTable> _lut;

    // The main actor
    vtkSmartPointer<vtkActor> _actor;

    // The renderer
    vtkSmartPointer<vtkRenderer> _renderer;

    // The render window
    vtkSmartPointer<vtkRenderWindow> _renderWindow;

    // The scalar bar that gives the viewer the mapping from color to
    // scalar value
    vtkSmartPointer<vtkScalarBarActor> _scalarBar;

    // Axes
    vtkSmartPointer<vtkAxesActor> _axesActor;

    // Help text popup
    vtkSmartPointer<vtkTextActor> helptextActor;
    vtkSmartPointer<vtkBalloonRepresentation> balloonRep;
    vtkSmartPointer<vtkBalloonWidget> balloonwidget;

  };

}

#endif
