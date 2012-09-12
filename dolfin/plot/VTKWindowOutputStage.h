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
// Last changed: 2012-09-12

#ifndef __VTK_WINDOW_OUTPUT_STAGE_H
#define __VTK_WINDOW_OUTPUT_STAGE_H

#include <boost/scoped_ptr.hpp>
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

    vtkSmartPointer<vtkAxesActor> _axesActor;

    vtkSmartPointer<vtkTextActor> helptextActor;
    vtkSmartPointer<vtkBalloonRepresentation> balloonRep;
    vtkSmartPointer<vtkBalloonWidget> balloonwidget;

#ifdef HAS_QVTK
    boost::scoped_ptr<QVTKWidget> widget;
#endif

  public:
    VTKWindowOutputStage();

    void init(VTKPlotter *parent, const Parameters &parameters);

    vtkRenderWindowInteractor* get_interactor();

    vtkSmartPointer<vtkRenderer> get_renderer();

    void scale_points_lines(double factor);

    void set_helptext(std::string text);

    void set_window_title(std::string title);

    std::string get_window_title();

    void close_window();

    bool resurrect_window();

    void start_interaction(bool enter_eventloop=true);

    void stop_interaction();

    void write_png(std::string filename);

    void write_pdf(std::string filename);

    vtkCamera* get_camera();

    void reset_camera();

    void set_scalar_range(double *range);

    void cycle_representation(int new_rep=0);

    void render();

    void get_window_size(int& width, int& height);

    void get_screen_size(int& width, int& height);

    void place_window(int x, int y);

    bool add_viewprop(vtkSmartPointer<vtkProp> prop);

    void set_input(vtkSmartPointer<vtkAlgorithmOutput> output);

    void set_translucent(bool onoff, uint topo_dim=3, uint geom_dim=3);

    ~VTKWindowOutputStage();

  };

} // namespace dolfin

#endif
