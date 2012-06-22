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
// Last changed: 2012-06-21

#ifdef HAS_VTK

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
VTKPlotter::VTKPlotter(boost::shared_ptr<const Mesh> mesh) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(new VTKPlottableMesh(mesh))),
  _frame_counter(0),
  _id(mesh->id())
{
  parameters = default_mesh_parameters();
  set_title(mesh->name(), mesh->label());
  init_pipeline();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const Function> function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
        new VTKPlottableGenericFunction(function))),
  _frame_counter(0),
  _id(function->id())
{
  parameters = default_parameters();
  set_title(function->name(), function->label());
  init_pipeline();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const ExpressionWrapper> expression) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
        new VTKPlottableGenericFunction(expression->expression(),
          expression->mesh()))),
  _frame_counter(0),
  _id(expression->id())
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
  _frame_counter(0),
  _id(expression->id())
{
  parameters = default_parameters();
  set_title(expression->name(), expression->label());
  init_pipeline();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const DirichletBC> bc) :
  _frame_counter(0),
  _id(bc->id())
{
  boost::shared_ptr<Function> f(new Function(bc->function_space()));
  bc->apply(*f->vector());
  _plottable = boost::shared_ptr<VTKPlottableMesh>(
      new VTKPlottableGenericFunction(f));

  parameters = default_parameters();
  set_title(bc->name(), bc->label());
  init_pipeline();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<uint> > mesh_function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
        new VTKPlottableMeshFunction<uint>(mesh_function))),
  _frame_counter(0),
  _id(mesh_function->id())
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
  _frame_counter(0),
  _id(mesh_function->id())
{
  parameters = default_parameters();
  set_title(mesh_function->name(), mesh_function->label());
  init_pipeline();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<double> > mesh_function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
        new VTKPlottableMeshFunction<double>(mesh_function))),
  _frame_counter(0),
  _id(mesh_function->id())
{
  parameters = default_parameters();
  set_title(mesh_function->name(), mesh_function->label());
  init_pipeline();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<bool> > mesh_function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
        new VTKPlottableMeshFunction<bool>(mesh_function))),
  _frame_counter(0),
  _id(mesh_function->id())
{
  parameters = default_parameters();
  set_title(mesh_function->name(), mesh_function->label());
  init_pipeline();
}
//----------------------------------------------------------------------------
VTKPlotter::~VTKPlotter()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKPlotter::plot()
{
  // Abort if DOLFIN_NOPLOT is set to a nonzero value
  char *noplot;
  noplot = getenv("DOLFIN_NOPLOT");
  if (noplot != NULL) {
    if (strcmp(noplot, "0") != 0 && strcmp(noplot, "") != 0) {
      warning("Environment variable DOLFIN_NOPLOT set: Plotting disabled.");
      return;
    }
  }

  // The plotting starts

  // Process some parameters
  if (parameters["wireframe"]) {
    _actor->GetProperty()->SetRepresentationToWireframe();
  }
  if (parameters["scalarbar"]) {
    _renderer->AddActor(_scalarBar);
  }

  _renderWindow->SetWindowName(std::string(parameters["title"]).c_str());

  // Update the plottable data
  _plottable->update(parameters);
  
  // If this is the first render of this plot and/or the rescale parameter is
  // set, we read get the min/max values of the data and process them 
  if (_frame_counter == 0 || parameters["rescale"]) {
    double range[2];
    
    _plottable->update_range(range);

    _lut->SetRange(range);
    _lut->Build();

    _mapper->SetScalarRange(range);

  }

  // Set the mapper's connection on each plot. This must be done since the
  // visualization parameters may have changed since the last frame, and 
  // the input may hence also have changed
  _mapper->SetInputConnection(_plottable->get_output());

  _renderWindow->Render();

  _frame_counter++;

  if (parameters["interactive"]) {
    interactive();
  }
}
//----------------------------------------------------------------------------
void VTKPlotter::interactive()
{
  // Add keypress callback function
  _interactor->AddObserver(vtkCommand::KeyPressEvent, this, 
      &VTKPlotter::keypressCallback); 

  // These must be declared outside the if test to not go out of scope
  // before the interaction starts
  vtkSmartPointer<vtkTextActor> helptextActor =
    vtkSmartPointer<vtkTextActor>::New();
  vtkSmartPointer<vtkBalloonRepresentation> balloonRep = 
    vtkSmartPointer<vtkBalloonRepresentation>::New();
  vtkSmartPointer<vtkBalloonWidget> balloonwidget =
    vtkSmartPointer<vtkBalloonWidget>::New();

  if (parameters["helptext"]) {
    // Add help text actor
    helptextActor->SetPosition(10,10);
    helptextActor->SetInput("Help ");
    helptextActor->GetTextProperty()->SetColor(0.0, 0.0, 0.0);
    helptextActor->GetTextProperty()->SetFontSize(20);
    _renderer->AddActor2D(helptextActor);

    // Set up the representation for the hover-over help text box
    balloonRep->SetOffset(5,5);
    balloonRep->GetTextProperty()->SetFontSize(18);
    balloonRep->GetTextProperty()->BoldOff();
    balloonRep->GetFrameProperty()->SetOpacity(0.7);

    // Set up the actual widget that makes the help text pop up
    balloonwidget->SetInteractor(_interactor);
    balloonwidget->SetRepresentation(balloonRep);
    balloonwidget->AddBalloon(helptextActor,
        get_helptext().c_str(),NULL);
    _renderWindow->Render();
    balloonwidget->EnabledOn();
  }

  // Initialize and start the mouse interaction
  _interactor->Initialize();
  _interactor->Start();
}
//----------------------------------------------------------------------------
void VTKPlotter::init_pipeline()
{
  // We first initialize the part of the pipeline that the plotter controls.
  // This is the part from the Poly data mapper and out, including actor, 
  // renderer, renderwindow and interaction. It also takes care of the scalar
  // bar and other decorations. 

  // Initialize objects
  _scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
  _lut = vtkSmartPointer<vtkLookupTable>::New();
  _mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  _actor = vtkSmartPointer<vtkActor>::New();
  _renderer = vtkSmartPointer<vtkRenderer>::New();
  _renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  _interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

  // Connect the parts
  _mapper->SetLookupTable(_lut);
  _scalarBar->SetLookupTable(_lut);
  _actor->SetMapper(_mapper);
  _renderer->AddActor(_actor);
  _renderWindow->AddRenderer(_renderer);

  // Set up interactorstyle and connect interactor
  vtkSmartPointer<vtkInteractorStyleTrackballCamera> style =
    vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
  _interactor->SetRenderWindow(_renderWindow);
  _interactor->SetInteractorStyle(style);

  // Set some properties that affect the look of things
  _renderer->SetBackground(1, 1, 1);
  _actor->GetProperty()->SetColor(0, 0, 1); //Only used for meshes
  _renderWindow->SetSize(parameters["window_width"],
      parameters["window_height"]);
  _scalarBar->SetTextPositionToPrecedeScalarBar();

  // Set the look of scalar bar labels
  vtkSmartPointer<vtkTextProperty> labelprop =
    _scalarBar->GetLabelTextProperty();
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
  text << "\t I/O: Turn vertex indices on/off\n";
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

  switch (iren->GetKeyCode()) {
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
      // Check if actor is present. If not, get from plottable. If it is, turn on visibility
      vtkSmartPointer<vtkActor2D> labels = _plottable->get_vertex_label_actor();
      if (_renderer->HasViewProp(labels)) {
        labels->VisibilityOn();
      } else {
        _renderer->AddActor(labels);
      }
      _renderWindow->Render();
      break;
    }
    case 'o': 
    {
      // Check if actor is present. If not, ignore. If it is, turn off visibility
      vtkSmartPointer<vtkActor2D> labels = _plottable->get_vertex_label_actor();
      if (_renderer->HasViewProp(labels)) {
        labels->VisibilityOff();
      }
      _renderWindow->Render();
      break;
    }
    default:
      break;
  }
}
//----------------------------------------------------------------------------
void VTKPlotter::hardcopy(std::string filename)
{
  dolfin_assert(_renderWindow);

  info("Saving plot to file: %s.png", filename.c_str());

  // FIXME: Remove help-text-actor before hardcopying.

  // Create window to image filter and PNG writer
  vtkSmartPointer<vtkWindowToImageFilter> w2i = 
    vtkSmartPointer<vtkWindowToImageFilter>::New();
  vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();

  w2i->SetInput(_renderWindow);
  w2i->Update();
  writer->SetInputConnection(w2i->GetOutputPort());
  writer->SetFileName((filename + ".png").c_str());
  _renderWindow->Render();
  writer->Modified();
  writer->Write();
}
//----------------------------------------------------------------------------
void VTKPlotter::get_window_size(int& width, int& height)
{
  dolfin_assert(_interactor);
  _interactor->GetSize(width, height);
}
//----------------------------------------------------------------------------
void VTKPlotter::set_window_position(int x, int y)
{
  dolfin_assert(_renderWindow);
  _renderWindow->SetPosition(x, y);
}
#endif
