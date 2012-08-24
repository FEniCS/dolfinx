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
// Modified by Joachim B Haga 2012
//
// First added:  2012-05-23
// Last changed: 2012-08-23


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
#include <vtkObjectFactory.h>

#include <boost/filesystem.hpp>


using namespace dolfin;

//----------------------------------------------------------------------------
namespace dolfin
{
//----------------------------------------------------------------------------
  class PrivateVTKInteractorStyle : public vtkInteractorStyleTrackballCamera
  {
    // Create a new style instead of observer callbacks, so that we can
    // intercept keypresses (like q/e) reliably.
  public:

    PrivateVTKInteractorStyle() : _plotter(NULL) {}

    static PrivateVTKInteractorStyle* New();
    vtkTypeMacro(PrivateVTKInteractorStyle, vtkInteractorStyleTrackballCamera);

    virtual void OnKeyPress() {
      if (!_plotter->keypressCallback(Interactor->GetKeySym()))
        vtkInteractorStyleTrackballCamera::OnKeyPress();
    }

    // A reference to the parent plotter
    VTKPlotter *_plotter;
  };
  vtkStandardNewMacro(PrivateVTKInteractorStyle);
//----------------------------------------------------------------------------
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


    void init(VTKPlotter *parent, const Parameters &parameters)
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
      vtkSmartPointer<PrivateVTKInteractorStyle> style =
        vtkSmartPointer<PrivateVTKInteractorStyle>::New();
      style->_plotter = parent;
      _interactor->SetRenderWindow(_renderWindow);
      _interactor->SetInteractorStyle(style);

      // Set some properties that affect the look of things
      _renderer->SetBackground(1, 1, 1);
      _lut->SetNanColor(0.0, 0.0, 0.0, 0.1);
      _actor->GetProperty()->SetColor(0, 0, 1); //Only used for meshes
      _actor->GetProperty()->SetPointSize(3);   // should be parameter?
      //_actor->GetProperty()->SetLineWidth(1);

      // Set window stuff
      _renderWindow->SetSize(parameters["window_width"], parameters["window_height"]);
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
//----------------------------------------------------------------------------
} // namespace dolfin
//----------------------------------------------------------------------------
namespace {
  void round_significant_digits(double &x, double (*rounding)(double), int num_significant_digits)
  {
    if (x != 0.0)
    {
      const int num_digits = std::log10(std::abs(x))+1;
      const double reduction_factor = std::pow(10, num_digits-num_significant_digits);
      x = rounding(x/reduction_factor)*reduction_factor;
    }
  }
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const Mesh> mesh) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(new VTKPlottableMesh(mesh))),
  vtk_pipeline(new PrivateVTKPipeline()),
  _frame_counter(0),
  _key(to_key(*mesh)),
  _toggle_vertex_labels(false)
{
  parameters = default_mesh_parameters();
  set_title_from(*mesh);
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const Function> function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
        new VTKPlottableGenericFunction(function))),
  vtk_pipeline(new PrivateVTKPipeline()),
  _frame_counter(0),
  _key(to_key(*function)),
  _toggle_vertex_labels(false)
{
  parameters = default_parameters();
  set_title_from(*function);
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const Expression> expression,
    boost::shared_ptr<const Mesh> mesh) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
    new VTKPlottableGenericFunction(expression, mesh))),
  vtk_pipeline(new PrivateVTKPipeline()),
  _frame_counter(0),
  _key(to_key(*expression)),
  _toggle_vertex_labels(false)
{
  parameters = default_parameters();
  set_title_from(*expression);
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const ExpressionWrapper> wrapper) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
    new VTKPlottableGenericFunction(wrapper->expression(), wrapper->mesh()))),
  vtk_pipeline(new PrivateVTKPipeline()),
  _frame_counter(0),
  _key(to_key(*wrapper->expression())),
  _toggle_vertex_labels(false)
{
  parameters = default_parameters();
  set_title_from(*wrapper->expression());
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const DirichletBC> bc) :
  vtk_pipeline(new PrivateVTKPipeline()),
  _frame_counter(0),
  _key(to_key(*bc)),
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
  set_title_from(*bc);
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<uint> > mesh_function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
        new VTKPlottableMeshFunction<uint>(mesh_function))),
  vtk_pipeline(new PrivateVTKPipeline()),
  _frame_counter(0),
  _key(to_key(*mesh_function)),
  _toggle_vertex_labels(false)
{
  // FIXME: A different lookuptable should be set when plotting MeshFunctions
  parameters = default_parameters();
  set_title_from(*mesh_function);
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<int> > mesh_function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
    new VTKPlottableMeshFunction<int>(mesh_function))),
  vtk_pipeline(new PrivateVTKPipeline()),
  _frame_counter(0),
  _key(to_key(*mesh_function)),
  _toggle_vertex_labels(false)
{
  parameters = default_parameters();
  set_title_from(*mesh_function);
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<double> > mesh_function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
    new VTKPlottableMeshFunction<double>(mesh_function))),
  vtk_pipeline(new PrivateVTKPipeline()),
  _frame_counter(0),
  _key(to_key(*mesh_function)),
  _toggle_vertex_labels(false)
{
  parameters = default_parameters();
  set_title_from(*mesh_function);
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<bool> > mesh_function) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
    new VTKPlottableMeshFunction<bool>(mesh_function))),
  vtk_pipeline(new PrivateVTKPipeline()),
  _frame_counter(0),
  _key(to_key(*mesh_function)),
  _toggle_vertex_labels(false)
{
  parameters = default_parameters();
  set_title_from(*mesh_function);
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::~VTKPlotter()
{
  for (std::list<VTKPlotter*>::iterator it = all_plotters->begin(); it != all_plotters->end(); it++)
  {
    if (*it == this)
    {
      all_plotters->erase(it);
      return;
    }
  }
  // Plotter not found. This point should never be reached.
  dolfin_assert(false);
}
//----------------------------------------------------------------------------
void VTKPlotter::plot(boost::shared_ptr<const Variable> variable)
{
  // Abort if DOLFIN_NOPLOT is set to a nonzero value.
  if (no_plot)
  {
    warning("Environment variable DOLFIN_NOPLOT set: Plotting disabled.");
    return;
  }

  update(variable);

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

  // Create plotter pool if needed (ie. this is the first plotter object)
  if (all_plotters.get() == NULL)
  {
    log(TRACE, "Creating global VTKPlotter pool");
    all_plotters.reset(new std::list<VTKPlotter*>);
  }

  // Add plotter to pool
  all_plotters->push_back(this);

  // Add a local shared_ptr to the pool. See comment in VTKPlotter.h
  all_plotters_local_copy = all_plotters;
  log(TRACE, "Size of plotter pool is %d.", all_plotters->size());

  // Initialise PrivateVTKPipeline
  dolfin_assert(vtk_pipeline);
  vtk_pipeline->init(this, parameters);

  // Adjust window position to not completely overlap previous plots
  dolfin::uint num_old_plots = VTKPlotter::all_plotters->size()-1;
  int width, height;
  get_window_size(width, height);

  // FIXME: This approach might need some tweaking. It tiles the winows in a
  // 2 x 2 pattern on the screen. Might not look good with more than 4 plot
  // windows
  set_window_position((num_old_plots%2)*width, (num_old_plots/2)*height);

  // We first initialize the part of the pipeline that the plotter controls.
  // This is the part from the Poly data mapper and out, including actor,
  // renderer, renderwindow and interaction. It also takes care of the scalar
  // bar and other decorations.

  // Let the plottable initialize its part of the pipeline
  _plottable->init_pipeline();
}
//----------------------------------------------------------------------------
const std::string& VTKPlotter::key() const
{
  return _key;
}
//----------------------------------------------------------------------------
void VTKPlotter::set_key(std::string key)
{
  _key = key;
}
//----------------------------------------------------------------------------
std::string VTKPlotter::to_key(const Variable &var)
{
  std::stringstream s;
  s << var.id() << "@@";
  return s.str();
}
//----------------------------------------------------------------------------
void VTKPlotter::set_title_from(const Variable &variable)
{

  std::stringstream title;
  title << "Plot of \"" << variable.name() << "\"" << " (" << variable.label() << ")";
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
  text << "\t r: Reset zoom\n";
  text << "\t w: View figure as wireframe\n";
  text << "\t s: View figure with solid surface\n";
  text << "\t f: Fly to the point currently under the mouse pointer\n";
  text << "\t p: Add bounding box\n";
  text << "\t i: Toggle vertex indices on/off\n";
  text << "\t h: Save plot to file\n";
  text << "\t e/q: Exit\n";
  return text.str();
}
//----------------------------------------------------------------------------
bool VTKPlotter::keypressCallback(std::string key)
{
  // Save plot to file
  if (key == "h")
  {
    write_png();
    return true;
  }

  // Toggle vertex labels
  if (key == "i")
  {
    // Check if label actor is present. If not get from plottable. If it
    // is, toggle off
    vtkSmartPointer<vtkActor2D> labels = _plottable->get_vertex_label_actor();

    // Check for existence of labels
    if (!vtk_pipeline->_renderer->HasViewProp(labels))
      vtk_pipeline->_renderer->AddActor(labels);

    // Turn on or off dependent on present state
    _toggle_vertex_labels = !_toggle_vertex_labels;
    labels->SetVisibility(_toggle_vertex_labels);

    vtk_pipeline->_renderWindow->Render();
    return true;
  }

  // Something
  if (key == "Up")
  {
    std::cout << "Up arrow pressed." << std::endl;
  }

  // Not handled
  return false;
}
//----------------------------------------------------------------------------
void VTKPlotter::write_png(std::string filename)
{
  dolfin_assert(vtk_pipeline);
  dolfin_assert(vtk_pipeline->_renderWindow);

  if (filename.empty()) {
    // We construct a filename from the given prefix and static counter.
    // If a file with that filename exists, the counter is incremented
    // until a unique filename is found.
    std::stringstream filenamebuilder;
    filenamebuilder << std::string(parameters["prefix"]);
    filenamebuilder << hardcopy_counter;
    while (boost::filesystem::exists(filenamebuilder.str() + ".png")) {
      hardcopy_counter++;
      filenamebuilder.str("");
      filenamebuilder << std::string(parameters["prefix"]);
      filenamebuilder << hardcopy_counter;
    }
    filename = filenamebuilder.str();
  }

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
void VTKPlotter::update(boost::shared_ptr<const Variable> variable)
{
  if (!is_compatible(variable))
  {
    dolfin_error("VTKPlotter.cpp",
                 "plot()",
                 "The plottable is not compatible with the data");
  }

  // Process some parameters
  if (parameters["wireframe"])
    vtk_pipeline->_actor->GetProperty()->SetRepresentationToWireframe();

  if (parameters["scalarbar"])
    vtk_pipeline->_renderer->AddActor(vtk_pipeline->_scalarBar);

  vtk_pipeline->_renderWindow->SetWindowName(std::string(parameters["title"]).c_str());

  // Update the plottable data
  _plottable->update(variable, parameters, _frame_counter);

  // If this is the first render of this plot and/or the rescale parameter
  // is set, we read get the min/max values of the data and process them
  if (_frame_counter == 0 || parameters["rescale"])
  {
    double range[2];

    if (parameters["autorange"])
    {
      _plottable->update_range(range);

      // Round small values to zero
      const double diff = range[1]-range[0];
      if (diff != 0 && std::abs(range[0]/diff) < 1e-3)
        range[0] = 0;
      else if (diff != 0 && std::abs(range[1]/diff) < 1e-3)
        range[1] = 0;

      // Round endpoints to 2 significant digits (away from center)
      round_significant_digits(range[0], std::floor, 2);
      round_significant_digits(range[1], std::ceil,  2);
    }
    else
    {
      range[0] = parameters["range_min"];
      range[1] = parameters["range_max"];
    }

    vtk_pipeline->_mapper->SetScalarRange(range);
    // Not required, the mapper controls the range.
    //vtk_pipeline->_lut->SetRange(range);
    //vtk_pipeline->_lut->Build();
  }

  // Set the mapper's connection on each plot. This must be done since the
  // visualization parameters may have changed since the last frame, and
  // the input may hence also have changed
  vtk_pipeline->_mapper->SetInputConnection(_plottable->get_output());
}
//----------------------------------------------------------------------------
bool VTKPlotter::is_compatible(boost::shared_ptr<const Variable> variable) const
{
  return (!variable || _plottable->is_compatible(*variable));
}
//----------------------------------------------------------------------------
void VTKPlotter::all_interactive()
{
  if (all_plotters.get() == NULL || all_plotters->size() == 0)
    warning("No plots have been shown yet. Ignoring call to interactive().");
  else
  {
    // Prepare interactiveness on every plotter
    std::list<VTKPlotter*>::iterator it;
    for (it = all_plotters->begin(); it != all_plotters->end(); it++)
      (*it)->interactive(false);

    // Start the vtk eventloop on one of the plotter objects
    (*all_plotters->begin())->start_eventloop();
  }
}


#else

// Implement dummy version of class VTKPlotter even if VTK is not present.


#include "VTKPlotter.h"
namespace dolfin { class PrivateVTKPipeline{}; }

using namespace dolfin;

VTKPlotter::VTKPlotter(boost::shared_ptr<const Mesh> mesh) : _key(to_key(*mesh))                                    { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const Function> function) : _key(to_key(*function))                        { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const Expression> expression,
		       boost::shared_ptr<const Mesh> mesh) : _key(to_key(*expression))                              { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const DirichletBC> bc) : _key(to_key(*bc))                                 { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<uint> > mesh_function) : _key(to_key(*mesh_function))   { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<int> > mesh_function) : _key(to_key(*mesh_function))    { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<double> > mesh_function) : _key(to_key(*mesh_function)) { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const MeshFunction<bool> > mesh_function) : _key(to_key(*mesh_function))   { init(); }
VTKPlotter::~VTKPlotter(){}

// (Ab)use init() to issue a warning.
// We also need to initialize the parameter set to avoid tons of warning
// when running the tests without VTK.

void VTKPlotter::init()
{
  parameters = default_parameters();
  warning("Plotting not available. DOLFIN has been compiled without VTK support.");
}

void VTKPlotter::update(boost::shared_ptr<const Variable>) {}


void VTKPlotter::plot               (boost::shared_ptr<const Variable>) {}
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
void VTKPlotter::add_polygon        (const Array<double>&){}

void VTKPlotter::all_interactive() {}

#endif

// Define the static members
boost::shared_ptr<std::list<VTKPlotter*> > VTKPlotter::all_plotters;
int VTKPlotter::hardcopy_counter = 0;
