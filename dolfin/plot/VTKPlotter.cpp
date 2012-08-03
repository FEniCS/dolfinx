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
// Modified by Benjamin Kehlet 2012
// Modified by Garth N. Wells 2012
//
// First added:  2012-05-23
// Last changed: 2012-07-26


#include <dolfin/common/Array.h>
#include <dolfin/common/Timer.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include "ExpressionWrapper.h"
#include "VTKPlottableGenericFunction.h"
#include "VTKPlottableMesh.h"
#include "VTKPlottableMeshFunction.h"
#include "VTKPlotter.h"

#ifdef HAS_VTK

#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkLookupTable.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkScalarBarActor.h>
#include <vtkTextProperty.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkTextActor.h>
#include <vtkBalloonRepresentation.h>
#include <vtkBalloonWidget.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkCommand.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkPoints.h>
#include <vtkPolyLine.h>
#include <vtkCylinderSource.h>

#include <boost/filesystem.hpp>


using namespace dolfin;

//----------------------------------------------------------------------------
namespace dolfin
{
  class PrivateVTKPipeline
  {
  public:

    // The poly data mapper
    vtkSmartPointer<vtkPolyDataMapper> _mapper;

    // The lookup table
    vtkSmartPointer<vtkLookupTable> _lut;

    // The main actor
    vtkSmartPointer<vtkActor> _actor;

    // The actor for polygons
    vtkSmartPointer<vtkActor> polygon_actor;

    // The renderer
    vtkSmartPointer<vtkRenderer> _renderer;

    // The render window
    vtkSmartPointer<vtkRenderWindow> _renderWindow;

    // The render window interactor for interactive sessions
    vtkSmartPointer<vtkRenderWindowInteractor> _interactor;

    // The scalar bar that gives the viewer the mapping from color to
    // scalar value
    vtkSmartPointer<vtkScalarBarActor> _scalarBar;

    // Note: VTK (current 5.6.1) seems to very picky about the order
    // of destruction. It seg faults if the objects are destroyed
    // first (probably before the renderer).
    vtkSmartPointer<vtkTextActor> helptextActor;
    vtkSmartPointer<vtkBalloonRepresentation> balloonRep;
    vtkSmartPointer<vtkBalloonWidget> balloonwidget;


    PrivateVTKPipeline()
    {
      // Initialize objects
      _scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
      _lut = vtkSmartPointer<vtkLookupTable>::New();
      _mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
      _actor = vtkSmartPointer<vtkActor>::New();
      polygon_actor = vtkSmartPointer<vtkActor>::New();
      helptextActor = vtkSmartPointer<vtkTextActor>::New();
      balloonRep = vtkSmartPointer<vtkBalloonRepresentation>::New();
      balloonwidget = vtkSmartPointer<vtkBalloonWidget>::New();

      _renderer = vtkSmartPointer<vtkRenderer>::New();
      _renderWindow = vtkSmartPointer<vtkRenderWindow>::New();

      _interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

      // Connect the parts
      _mapper->SetLookupTable(_lut);
      _scalarBar->SetLookupTable(_lut);
      _actor->SetMapper(_mapper);
      _renderer->AddActor(_actor);
      _renderer->AddActor(polygon_actor);
      _renderWindow->AddRenderer(_renderer);

      // Set up interactorstyle and connect interactor
      vtkSmartPointer<vtkInteractorStyleTrackballCamera> style
	= vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
      _interactor->SetRenderWindow(_renderWindow);
      _interactor->SetInteractorStyle(style);

      // Set some properties that affect the look of things
      _renderer->SetBackground(1, 1, 1);
      _actor->GetProperty()->SetColor(0, 0, 1); //Only used for meshes

      // FIXME: Take this as parameter
      _renderWindow->SetSize(600, 400);
      _scalarBar->SetTextPositionToPrecedeScalarBar();

      // Set the look of scalar bar labels
      vtkSmartPointer<vtkTextProperty> labelprop
	= _scalarBar->GetLabelTextProperty();
      labelprop->SetColor(0, 0, 0);
      labelprop->SetFontSize(20);
      labelprop->ItalicOff();
      labelprop->BoldOff();
    }
  };
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const Mesh> mesh) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(new VTKPlottableMesh(mesh))),
  vtk_pipeline(new PrivateVTKPipeline),
  _frame_counter(0),
  _id(mesh->id()),
  _toggle_vertex_labels(false)
{

  parameters = default_mesh_parameters();
  set_title(mesh->name(), mesh->label());
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const Function> function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
        new VTKPlottableGenericFunction(function))),
  vtk_pipeline(new PrivateVTKPipeline),
  _frame_counter(0),
  _id(function->id()),
  _toggle_vertex_labels(false)
{
  parameters = default_parameters();
  set_title(function->name(), function->label());
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const ExpressionWrapper> expression) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
    new VTKPlottableGenericFunction(expression->expression(), expression->mesh()))),
  vtk_pipeline(new PrivateVTKPipeline),
  _frame_counter(0),
  _id(expression->id()),
  _toggle_vertex_labels(false)
{
  parameters = default_parameters();
  set_title(expression->expression()->name(),
            expression->expression()->label());
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const Expression> expression,
    boost::shared_ptr<const Mesh> mesh) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
    new VTKPlottableGenericFunction(expression, mesh))),
  vtk_pipeline(new PrivateVTKPipeline),
  _frame_counter(0),
  _id(expression->id()),
  _toggle_vertex_labels(false)
{
  parameters = default_parameters();
  set_title(expression->name(), expression->label());
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const DirichletBC> bc) :
  vtk_pipeline(new PrivateVTKPipeline),
  _frame_counter(0),
  _id(bc->id()),
  _toggle_vertex_labels(false)
{
  dolfin_error("VTKPlotter.cpp",
               "create plotter for Dirichlet B.C.",
               "Plotting of boundary conditions is not yet implemented");

  // FIXME: There is something wrong with the below code. The function is not
  // plotted, only an empty plotting window is shown.
  boost::shared_ptr<Function> f(new Function(bc->function_space()));
  bc->apply(*f->vector());
  _plottable
    = boost::shared_ptr<VTKPlottableMesh>(new VTKPlottableGenericFunction(f));

  parameters = default_parameters();
  set_title(bc->name(), bc->label());
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<uint> > mesh_function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
        new VTKPlottableMeshFunction<uint>(mesh_function))),
  vtk_pipeline(new PrivateVTKPipeline),
  _frame_counter(0),
  _id(mesh_function->id()),
  _toggle_vertex_labels(false)
{
  // FIXME: A different lookuptable should be set when plotting MeshFunctions
  parameters = default_parameters();
  set_title(mesh_function->name(), mesh_function->label());
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<int> > mesh_function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
    new VTKPlottableMeshFunction<int>(mesh_function))),
  vtk_pipeline(new PrivateVTKPipeline),
  _frame_counter(0),
  _id(mesh_function->id()),
  _toggle_vertex_labels(false)
{
  parameters = default_parameters();
  set_title(mesh_function->name(), mesh_function->label());
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<double> > mesh_function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
    new VTKPlottableMeshFunction<double>(mesh_function))),
  vtk_pipeline(new PrivateVTKPipeline),
  _frame_counter(0),
  _id(mesh_function->id()),
  _toggle_vertex_labels(false)
{
  parameters = default_parameters();
  set_title(mesh_function->name(), mesh_function->label());
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<bool> > mesh_function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
    new VTKPlottableMeshFunction<bool>(mesh_function))),
  vtk_pipeline(new PrivateVTKPipeline),
  _frame_counter(0),
  _id(mesh_function->id()),
  _toggle_vertex_labels(false)
{
  parameters = default_parameters();
  set_title(mesh_function->name(), mesh_function->label());
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::~VTKPlotter()
{
  for (std::list<VTKPlotter*>::iterator it = plotter_cache.begin(); it != plotter_cache.end(); it++)
  {
    if (*it == this)
    {
      plotter_cache.erase(it);
      return;
    }
  }
  // Plotter not found. This point should never be reached.
  dolfin_assert(false);
}
//----------------------------------------------------------------------------
void VTKPlotter::plot()
{
  // Abort if DOLFIN_NOPLOT is set to a nonzero value.
  if (no_plot)
  {
    warning("Environment variable DOLFIN_NOPLOT set: Plotting disabled.");
    return;
  }

  update();

  vtk_pipeline->_renderWindow->Render();

  _frame_counter++;

  if (parameters["interactive"])
    interactive();
}
//----------------------------------------------------------------------------
void VTKPlotter::interactive(bool enter_eventloop)
{

  // Abort if DOLFIN_NOPLOT is set to a nonzero value
  if (no_plot)
    return;

  dolfin_assert(vtk_pipeline);

  // Add keypress callback function
  vtk_pipeline->_interactor->AddObserver(vtkCommand::KeyPressEvent, this,
                                         &VTKPlotter::keypressCallback);

  // These must be declared outside the if test to not go out of scope
  // before the interaction starts


  if (parameters["helptext"])
  {
    // Add help text actor
    vtk_pipeline->helptextActor->SetPosition(10,10);
    vtk_pipeline->helptextActor->SetInput("Help ");
    vtk_pipeline->helptextActor->GetTextProperty()->SetColor(0.0, 0.0, 0.0);
    vtk_pipeline->helptextActor->GetTextProperty()->SetFontSize(20);
    vtk_pipeline->_renderer->AddActor2D(vtk_pipeline->helptextActor);

    // Set up the representation for the hover-over help text box
    vtk_pipeline->balloonRep->SetOffset(5,5);
    vtk_pipeline->balloonRep->GetTextProperty()->SetFontSize(18);
    vtk_pipeline->balloonRep->GetTextProperty()->BoldOff();
    vtk_pipeline->balloonRep->GetFrameProperty()->SetOpacity(0.7);

    // Set up the actual widget that makes the help text pop up
    vtk_pipeline->balloonwidget->SetInteractor(vtk_pipeline->_interactor);
    vtk_pipeline->balloonwidget->SetRepresentation(vtk_pipeline->balloonRep);
    vtk_pipeline->balloonwidget->AddBalloon(vtk_pipeline->helptextActor,
                              get_helptext().c_str(),NULL);
    vtk_pipeline->balloonwidget->EnabledOn();
  }

  // Initialize and start the mouse interaction
  vtk_pipeline->_interactor->Initialize();

  vtk_pipeline->_renderWindow->Render();

  if (enter_eventloop)
    start_eventloop();
}
//----------------------------------------------------------------------------
void VTKPlotter::start_eventloop()
{
  if (!no_plot)
    vtk_pipeline->_interactor->Start();
}
//----------------------------------------------------------------------------
void VTKPlotter::init()
{
  // Check if environment variable DOLFIN_NOPLOT is set to a nonzero value
  {
    char *noplot_env;
    noplot_env = getenv("DOLFIN_NOPLOT");
    no_plot = (noplot_env != NULL && strcmp(noplot_env, "0") != 0 && strcmp(noplot_env, "") != 0);
  }

  // Adjust window position to not completely overlap previous plots
  dolfin::uint num_old_plots = VTKPlotter::plotter_cache.size();
  int width, height;
  get_window_size(width, height);

  // FIXME: This approach might need some tweaking. It tiles the winows in a
  // 2 x 2 pattern on the screen. Might not look good with more than 4 plot
  // windows
  set_window_position((num_old_plots%2)*width, (num_old_plots/2)*height);

  // Add plotter to cache
  plotter_cache.push_back(this);
  log(TRACE, "Size of plotter cache is %d.", plotter_cache.size());

  // We first initialize the part of the pipeline that the plotter controls.
  // This is the part from the Poly data mapper and out, including actor,
  // renderer, renderwindow and interaction. It also takes care of the scalar
  // bar and other decorations.

  dolfin_assert(vtk_pipeline);

  // Let the plottable initialize its part of the pipeline
  _plottable->init_pipeline();

  // That's it for the initialization! Now we wait until the user wants to
  // plot something
}
//----------------------------------------------------------------------------
void VTKPlotter::set_title(const std::string& name, const std::string& label)
{
  std::stringstream title;
  title <<"Plot of \"" << name << "\" (" << label << ")";
  parameters["title"] =  title.str();
}
//----------------------------------------------------------------------------
std::string VTKPlotter::get_helptext()
{
  std::stringstream text;

  text << "Mouse control:\n";
  text << "\t Left mouse button: Rotate figure\n";
  text << "\t Right mouse button (or scroolwheel): Zoom \n";
  text << "\t Middle mouse button (or left+right): Translate figure\n\n";
  text << "Keyboard control:\n";
  text << "\t R: Reset zoom\n";
  text << "\t W: View figure as wireframe\n";
  text << "\t S: View figure with solid surface\n";
  text << "\t F: Fly to the point currently under the mouse pointer\n";
  text << "\t P: Add bounding box\n";
  text << "\t I: Toggle vertex indices on/off\n";
  text << "\t H: Save plot to file\n";
  text << "\t E/Q: Exit\n";
  return text.str();
}
//----------------------------------------------------------------------------
void VTKPlotter::keypressCallback(vtkObject* caller,
                                  long unsigned int eventId,
                                  void* callData)
{
  vtkSmartPointer<vtkRenderWindowInteractor> iren
    = static_cast<vtkRenderWindowInteractor*>(caller);

  switch (iren->GetKeyCode())
  {
    // Save plot to file
    case 'h':
    {
      // We construct a filename from the given prefix and static counter.
      // If a file with that filename exists, the counter is incremented
      // until a unique filename is found. That filename is then passed
      // to the hardcopy function.
      std::stringstream filename;
      filename << std::string(parameters["prefix"]);
      filename << hardcopy_counter;
      while (boost::filesystem::exists(filename.str() + ".png")) {
        hardcopy_counter++;
        filename.str("");
        filename << std::string(parameters["prefix"]);
        filename << hardcopy_counter;
      }
      write_png(filename.str());
      break;
    }
    case 'i':
    {
      // Check if label actor is present. If not get from plottable. If it
      // is, toggle off
      vtkSmartPointer<vtkActor2D> labels = _plottable->get_vertex_label_actor();

      // Check for excistance of labels
      if (!vtk_pipeline->_renderer->HasViewProp(labels))
        vtk_pipeline->_renderer->AddActor(labels);

      // Turn on or off dependent on present state
      if (_toggle_vertex_labels)
      {
        labels->VisibilityOff();
        _toggle_vertex_labels = false;
      }
      else
      {
        labels->VisibilityOn();
        _toggle_vertex_labels = true;
      }

      vtk_pipeline->_renderWindow->Render();
      break;
    }
    default:
      break;
  }
}
//----------------------------------------------------------------------------
void VTKPlotter::write_png(std::string filename)
{
  dolfin_assert(vtk_pipeline);
  dolfin_assert(vtk_pipeline->_renderWindow);

  info("Saving plot to file: %s.png", filename.c_str());

  update();

  // FIXME: Remove help-text-actor before hardcopying.

  // Create window to image filter and PNG writer
  vtkSmartPointer<vtkWindowToImageFilter> w2i =
    vtkSmartPointer<vtkWindowToImageFilter>::New();
  vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();

  w2i->SetInput(vtk_pipeline->_renderWindow);
  w2i->Update();
  writer->SetInputConnection(w2i->GetOutputPort());
  writer->SetFileName((filename + ".png").c_str());
  vtk_pipeline->_renderWindow->Render();
  writer->Modified();
  writer->Write();
}
//----------------------------------------------------------------------------
void VTKPlotter::get_window_size(int& width, int& height)
{
  dolfin_assert(vtk_pipeline);
  dolfin_assert(vtk_pipeline->_interactor);
  vtk_pipeline->_interactor->GetSize(width, height);
}
//----------------------------------------------------------------------------
void VTKPlotter::set_window_position(int x, int y)
{
  dolfin_assert(vtk_pipeline);
  dolfin_assert(vtk_pipeline->_renderWindow);
  vtk_pipeline->_renderWindow->SetPosition(x, y);
}
//----------------------------------------------------------------------------
void VTKPlotter::azimuth(double angle)
{
  vtk_pipeline->_renderer->GetActiveCamera()->Azimuth(angle);
}
//----------------------------------------------------------------------------
void VTKPlotter::elevate(double angle)
{
  vtk_pipeline->_renderer->GetActiveCamera()->Elevation(angle);
}
//----------------------------------------------------------------------------
void VTKPlotter::dolly(double value)
{
  vtk_pipeline->_renderer->GetActiveCamera()->Dolly(value);
}
//----------------------------------------------------------------------------
void VTKPlotter::set_viewangle(double angle)
{
  vtk_pipeline->_renderer->GetActiveCamera()->SetViewAngle(angle);
}
//----------------------------------------------------------------------------
void VTKPlotter::set_min_max(double min, double max)
{
  parameters["autorange"] = false;
  parameters["range_min"] = min;
  parameters["range_max"] = max;
}
//----------------------------------------------------------------------------
void VTKPlotter::add_polygon(const Array<double>& points)
{
  const dolfin::uint dim = _plottable->dim();

  if (points.size() % dim != 0)
    warning("VTKPlotter::add_polygon() : Size of array is not a multiple of %d", dim);

  const dolfin::uint numpoints = points.size()/dim;

  vtkSmartPointer<vtkPoints> vtk_points = vtkSmartPointer<vtkPoints>::New();
  vtk_points->SetNumberOfPoints(numpoints);

  double point[3];
  point[2] = 0.0;

  for (dolfin::uint i = 0; i < numpoints; i++)
  {
    for (dolfin::uint j = 0; j < dim; j++)
      point[j] = points[i*dim + j];

    vtk_points->InsertPoint(i, point);
  }

  vtkSmartPointer<vtkPolyLine> line = vtkSmartPointer<vtkPolyLine>::New();
  line->GetPointIds()->SetNumberOfIds(numpoints);

  for (dolfin::uint i = 0; i < numpoints; i++)
    line->GetPointIds()->SetId(i, i);

  vtkSmartPointer<vtkUnstructuredGrid> grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  grid->Allocate(1, 1);

  grid->InsertNextCell(line->GetCellType(), line->GetPointIds());
  grid->SetPoints(vtk_points);

  vtkSmartPointer<vtkGeometryFilter> extract = vtkSmartPointer<vtkGeometryFilter>::New();
  extract->SetInput(grid);

  vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(extract->GetOutputPort());

  vtk_pipeline->polygon_actor->SetMapper(mapper);

  mapper->SetInputConnection(extract->GetOutputPort());

  vtk_pipeline->polygon_actor->GetProperty()->SetColor(0, 0, 1);
  vtk_pipeline->polygon_actor->GetProperty()->SetLineWidth(1);
}
//----------------------------------------------------------------------------
void VTKPlotter::update()
{
  // Process some parameters
  if (parameters["wireframe"])
    vtk_pipeline->_actor->GetProperty()->SetRepresentationToWireframe();

  if (parameters["scalarbar"])
    vtk_pipeline->_renderer->AddActor(vtk_pipeline->_scalarBar);

  vtk_pipeline->_renderWindow->SetWindowName(std::string(parameters["title"]).c_str());

  // Update the plottable data
  _plottable->update(parameters, _frame_counter);

  // If this is the first render of this plot and/or the rescale parameter
  // is set, we read get the min/max values of the data and process them
  if (_frame_counter == 0 || parameters["rescale"])
  {
    double range[2];

    if (parameters["autorange"])
      _plottable->update_range(range);
    else
    {
      range[0] = parameters["range_min"];
      range[1] = parameters["range_max"];
    }

    vtk_pipeline->_lut->SetRange(range);
    vtk_pipeline->_lut->Build();
    vtk_pipeline->_mapper->SetScalarRange(range);
  }

  // Set the mapper's connection on each plot. This must be done since the
  // visualization parameters may have changed since the last frame, and
  // the input may hence also have changed
  vtk_pipeline->_mapper->SetInputConnection(_plottable->get_output());
}

void VTKPlotter::update(boost::shared_ptr<const Mesh> mesh){ update(); }
void VTKPlotter::update(boost::shared_ptr<const Function> function) { update(); }
void VTKPlotter::update(boost::shared_ptr<const ExpressionWrapper> expression) { update(); }
void VTKPlotter::update(boost::shared_ptr<const Expression> expression, boost::shared_ptr<const Mesh> mesh) { update(); }
void VTKPlotter::update(boost::shared_ptr<const DirichletBC> bc) { update(); }
void VTKPlotter::update(boost::shared_ptr<const MeshFunction<unsigned int> > mesh_function) { update(); }
void VTKPlotter::update(boost::shared_ptr<const MeshFunction<int> > mesh_function) { update(); }
void VTKPlotter::update(boost::shared_ptr<const MeshFunction<double> > mesh_function){ update(); }
void VTKPlotter::update(boost::shared_ptr<const MeshFunction<bool> > mesh_function){ update(); }

#else

// Implement dummy version of class VTKPlotter even if VTK is not present.


#include "VTKPlotter.h"
#include "ExpressionWrapper.h"
namespace dolfin { class PrivateVTKPipeline{}; }

using namespace dolfin;

VTKPlotter::VTKPlotter(boost::shared_ptr<const Mesh> mesh) : _id(mesh->id())                                    { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const Function> function) : _id(function->id())                        { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const ExpressionWrapper> expression) : _id(expression->id())           { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const Expression> expression,
		       boost::shared_ptr<const Mesh> mesh) : _id(expression->id())                                          { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const DirichletBC> bc) : _id(bc->id())                                 { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<uint> > mesh_function) : _id(mesh_function->id())   { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<int> > mesh_function) : _id(mesh_function->id())    { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<double> > mesh_function) : _id(mesh_function->id()) { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<bool> > mesh_function) : _id(mesh_function->id())   { init(); }
VTKPlotter::~VTKPlotter(){}

// (Ab)use init() to issue a warning.
// We also need to initialize the parameter set to avoid tons of warning
// when running the tests without VTK.

void VTKPlotter::init()
{
  parameters = default_parameters();
  warning("Plotting not available. DOLFIN has been compiled without VTK support.");
}

void VTKPlotter::update(){}
void VTKPlotter::update(boost::shared_ptr<const Mesh> mesh){}
void VTKPlotter::update(boost::shared_ptr<const Function> function){}
void VTKPlotter::update(boost::shared_ptr<const ExpressionWrapper> expression){}
void VTKPlotter::update(boost::shared_ptr<const Expression> expression, boost::shared_ptr<const Mesh> mesh){}
void VTKPlotter::update(boost::shared_ptr<const DirichletBC> bc){}
void VTKPlotter::update(boost::shared_ptr<const MeshFunction<unsigned int> > mesh_function){}
void VTKPlotter::update(boost::shared_ptr<const MeshFunction<int> > mesh_function){}
void VTKPlotter::update(boost::shared_ptr<const MeshFunction<double> > mesh_function){}
void VTKPlotter::update(boost::shared_ptr<const MeshFunction<bool> > mesh_function){}


void VTKPlotter::plot               () {}
void VTKPlotter::interactive        (bool ){}
void VTKPlotter::start_eventloop    (){}
void VTKPlotter::write_png          (std::string){}
void VTKPlotter::get_window_size    (int&, int&){}
void VTKPlotter::set_window_position(int, int){}
void VTKPlotter::azimuth            (double) {}
void VTKPlotter::elevate            (double){}
void VTKPlotter::dolly              (double){}
void VTKPlotter::set_viewangle      (double){}
void VTKPlotter::set_min_max        (double, double){}
void VTKPlotter::add_polygon(const Array<double>&){}

#endif

// Define the static members
std::list<VTKPlotter*> VTKPlotter::plotter_cache;
int VTKPlotter::hardcopy_counter = 0;
