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
// Last changed: 2012-09-10

#ifndef __VTK_WINDOW_OUTPUT_STAGE_H
#define __VTK_WINDOW_OUTPUT_STAGE_H

#include <boost/scoped_ptr.hpp>
#include <vtkSmartPointer.h>

// Forward declarations
class QVTKWidget;
class vtkActor2D;
class vtkActor;
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
namespace dolfin {
  class GenericVTKPlottable;
  class Parameters;
  class VTKPlotter;
}

namespace dolfin
{

  /// This class enables visualization of various DOLFIN entities.
  class VTKWindowOutputStage
  {

  protected:

    // The depth sorting filter
    vtkSmartPointer<vtkDepthSortPolyData> _depthSort;

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

    // The scalar bar that gives the viewer the mapping from color to
    // scalar value
    vtkSmartPointer<vtkScalarBarActor> _scalarBar;

    vtkSmartPointer<vtkAxesActor> _axesActor;

    vtkSmartPointer<vtkTextActor> helptextActor;
    vtkSmartPointer<vtkBalloonRepresentation> balloonRep;
    vtkSmartPointer<vtkBalloonWidget> balloonwidget;

#ifdef HAS_QT4
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

    void close_window();

    bool resurrect_window();

    void start_interaction(bool enter_eventloop=true);

    void stop_interaction();

    void write_png(std::string filename);

    vtkCamera* get_camera();

    void set_scalar_range(double *range);

    void cycle_representation(int new_rep=0);

    void render();

    void get_window_size(int& width, int& height);

    void get_screen_size(int& width, int& height);

    void place_window(int x, int y);

    bool add_actor(vtkSmartPointer<vtkActor> actor);

    bool add_actor(vtkSmartPointer<vtkActor2D> actor);

    void set_input(const GenericVTKPlottable &plottable, bool reset_camera);

    ~VTKWindowOutputStage();

  };

} // namespace dolfin

#endif
