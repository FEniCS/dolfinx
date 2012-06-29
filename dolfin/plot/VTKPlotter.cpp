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
//
// First added:  2012-05-23
// Last changed: 2012-06-30

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

#include <boost/filesystem.hpp>

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/common/Timer.h>
#include "ExpressionWrapper.h"
#include "VTKPlottableMesh.h"
#include "VTKPlottableGenericFunction.h"
#include "VTKPlottableMeshFunction.h"
#include "VTKPlotter.h"

using namespace dolfin;

// Define the static members
std::vector<boost::shared_ptr<VTKPlotter> > VTKPlotter::plotter_cache;
int VTKPlotter::hardcopy_counter = 0;

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

    // The renderer
    vtkSmartPointer<vtkRenderer> _renderer;

    // The render window
    vtkSmartPointer<vtkRenderWindow> _renderWindow;

    // The render window interactor for interactive sessions
    vtkSmartPointer<vtkRenderWindowInteractor> _interactor;

    // The scalar bar that gives the viewer the mapping from color to
    // scalar value
    vtkSmartPointer<vtkScalarBarActor> _scalarBar;

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
  init_pipeline();
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
  init_pipeline();
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
  init_pipeline();
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
  init_pipeline();
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
  init_pipeline();
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
  init_pipeline();
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
  init_pipeline();
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
  init_pipeline();
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
  init_pipeline();
}
//----------------------------------------------------------------------------
VTKPlotter::~VTKPlotter()
{
  delete vtk_pipeline;
}
//----------------------------------------------------------------------------
void VTKPlotter::plot()
{
  // Abort if DOLFIN_NOPLOT is set to a nonzero value
  char *noplot;
  noplot = getenv("DOLFIN_NOPLOT");
  if (noplot != NULL)
  {
    if (strcmp(noplot, "0") != 0 && strcmp(noplot, "") != 0)
    {
      warning("Environment variable DOLFIN_NOPLOT set: Plotting disabled.");
      return;
    }
  }

  // The plotting starts

  // Process some parameters
  if (parameters["wireframe"])
    vtk_pipeline->_actor->GetProperty()->SetRepresentationToWireframe();

  if (parameters["scalarbar"])
    vtk_pipeline->_renderer->AddActor(vtk_pipeline->_scalarBar);

  vtk_pipeline->_renderWindow->SetWindowName(std::string(parameters["title"]).c_str());

  // Update the plottable data
  _plottable->update(parameters);



  if (parameters["autorange"])
  {

    // If this is the first render of this plot and/or the rescale parameter is
    // set, we read get the min/max values of the data and process them
    if (_frame_counter == 0 || parameters["rescale"])
    {
      double range[2];
      _plottable->update_range(range);
      vtk_pipeline->_lut->SetRange(range);
      vtk_pipeline->_lut->Build();
      vtk_pipeline->_mapper->SetScalarRange(range);
    }
  }
  else
  {
    double range[2];
    range[0] = parameters["range_min"];
    range[1] = parameters["range_max"];
    vtk_pipeline->_lut->SetRange(range);
    vtk_pipeline->_lut->Build();
    vtk_pipeline->_mapper->SetScalarRange(range);
  }
  

  // Set the mapper's connection on each plot. This must be done since the
  // visualization parameters may have changed since the last frame, and
  // the input may hence also have changed
  vtk_pipeline->_mapper->SetInputConnection(_plottable->get_output());

  vtk_pipeline->_renderWindow->Render();

  _frame_counter++;

  if (parameters["interactive"])
    interactive();
}
//----------------------------------------------------------------------------
void VTKPlotter::interactive()
{
  // Add keypress callback function
  vtk_pipeline->_interactor->AddObserver(vtkCommand::KeyPressEvent, this,
                                         &VTKPlotter::keypressCallback);

  // These must be declared outside the if test to not go out of scope
  // before the interaction starts
  vtkSmartPointer<vtkTextActor> helptextActor =
    vtkSmartPointer<vtkTextActor>::New();
  vtkSmartPointer<vtkBalloonRepresentation> balloonRep =
    vtkSmartPointer<vtkBalloonRepresentation>::New();
  vtkSmartPointer<vtkBalloonWidget> balloonwidget =
    vtkSmartPointer<vtkBalloonWidget>::New();

  if (parameters["helptext"])
  {
    // Add help text actor
    helptextActor->SetPosition(10,10);
    helptextActor->SetInput("Help ");
    helptextActor->GetTextProperty()->SetColor(0.0, 0.0, 0.0);
    helptextActor->GetTextProperty()->SetFontSize(20);
    vtk_pipeline->_renderer->AddActor2D(helptextActor);

    // Set up the representation for the hover-over help text box
    balloonRep->SetOffset(5,5);
    balloonRep->GetTextProperty()->SetFontSize(18);
    balloonRep->GetTextProperty()->BoldOff();
    balloonRep->GetFrameProperty()->SetOpacity(0.7);

    // Set up the actual widget that makes the help text pop up
    balloonwidget->SetInteractor(vtk_pipeline->_interactor);
    balloonwidget->SetRepresentation(balloonRep);
    balloonwidget->AddBalloon(helptextActor,
                              get_helptext().c_str(),NULL);
    vtk_pipeline->_renderWindow->Render();
    balloonwidget->EnabledOn();
  }

  // Initialize and start the mouse interaction
  vtk_pipeline->_interactor->Initialize();
  vtk_pipeline->_interactor->Start();
}
//----------------------------------------------------------------------------
void VTKPlotter::init_pipeline()
{
  // We first initialize the part of the pipeline that the plotter controls.
  // This is the part from the Poly data mapper and out, including actor,
  // renderer, renderwindow and interaction. It also takes care of the scalar
  // bar and other decorations.

  // Initialize objects
  vtk_pipeline->_scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
  vtk_pipeline->_lut = vtkSmartPointer<vtkLookupTable>::New();
  vtk_pipeline->_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  vtk_pipeline->_actor = vtkSmartPointer<vtkActor>::New();
  vtk_pipeline->_renderer = vtkSmartPointer<vtkRenderer>::New();
  vtk_pipeline->_renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  vtk_pipeline->_interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

  // Connect the parts
  vtk_pipeline->_mapper->SetLookupTable(vtk_pipeline->_lut);
  vtk_pipeline->_scalarBar->SetLookupTable(vtk_pipeline->_lut);
  vtk_pipeline->_actor->SetMapper(vtk_pipeline->_mapper);
  vtk_pipeline->_renderer->AddActor(vtk_pipeline->_actor);
  vtk_pipeline->_renderWindow->AddRenderer(vtk_pipeline->_renderer);

  // Set up interactorstyle and connect interactor
  vtkSmartPointer<vtkInteractorStyleTrackballCamera> style =
    vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
  vtk_pipeline->_interactor->SetRenderWindow(vtk_pipeline->_renderWindow);
  vtk_pipeline->_interactor->SetInteractorStyle(style);

  // Set some properties that affect the look of things
  vtk_pipeline->_renderer->SetBackground(1, 1, 1);
  vtk_pipeline->_actor->GetProperty()->SetColor(0, 0, 1); //Only used for meshes
  vtk_pipeline->_renderWindow->SetSize(parameters["window_width"],
				       parameters["window_height"]);
  vtk_pipeline->_scalarBar->SetTextPositionToPrecedeScalarBar();

  // Set the look of scalar bar labels
  vtkSmartPointer<vtkTextProperty> labelprop =
    vtk_pipeline->_scalarBar->GetLabelTextProperty();
  labelprop->SetColor(0, 0, 0);
  labelprop->SetFontSize(20);
  labelprop->ItalicOff();
  labelprop->BoldOff();

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
  vtkSmartPointer<vtkRenderWindowInteractor> iren =
    static_cast<vtkRenderWindowInteractor*>(caller);

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
      hardcopy(filename.str());
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
void VTKPlotter::hardcopy(std::string filename)
{
  dolfin_assert(vtk_pipeline->_renderWindow);

  info("Saving plot to file: %s.png", filename.c_str());

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
  dolfin_assert(vtk_pipeline->_interactor);
  vtk_pipeline->_interactor->GetSize(width, height);
}
//----------------------------------------------------------------------------
void VTKPlotter::set_window_position(int x, int y)
{
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

#endif
