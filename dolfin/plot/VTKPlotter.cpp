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
// Last changed: 2015-11-10

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
#include "VTKPlotter.h"

#ifdef HAS_VTK

#include "VTKWindowOutputStage.h"
#include "VTKPlottableGenericFunction.h"
#include "VTKPlottableMesh.h"
#include "VTKPlottableMeshFunction.h"
#include "VTKPlottableDirichletBC.h"

#include <vtkSmartPointer.h>
#include <vtkCamera.h>
#include <vtkPolyLine.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkUnstructuredGrid.h>
#include <vtkGeometryFilter.h>
#include <vtkToolkits.h>
#include <vtkRenderWindowInteractor.h>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#ifdef foreach
#undef foreach
#endif
#define foreach BOOST_FOREACH

using namespace dolfin;

//----------------------------------------------------------------------------
namespace // anonymous
{
  void round_significant_digits(double &x, double (*rounding)(double),
                                int num_significant_digits)
  {
    if (x != 0.0)
    {
      const int num_digits = std::log10(std::abs(x))+1;
      const double reduction_factor
        = std::pow(10.0, num_digits-num_significant_digits);
      x = rounding(x/reduction_factor)*reduction_factor;
    }
  }
}
//----------------------------------------------------------------------------
namespace dolfin
{
  GenericVTKPlottable* CreateVTKPlottable(std::shared_ptr<const Variable> var)
  {
#define DISPATCH(T) do                                                  \
    {                                                                   \
      std::shared_ptr<const T > t = std::dynamic_pointer_cast<const T >(var); \
      if (t)                                                            \
        return CreateVTKPlottable(t);                                   \
    } while (0)

    DISPATCH(DirichletBC);
    DISPATCH(ExpressionWrapper);
    DISPATCH(Function);
    DISPATCH(Mesh);
    DISPATCH(MeshFunction<bool>);
    DISPATCH(MeshFunction<double>);
    //DISPATCH(MeshFunction<float>);
    DISPATCH(MeshFunction<int>);
    DISPATCH(MeshFunction<std::size_t>);

    if (dynamic_cast<const Expression*>(var.get()))
    {
      dolfin_error("VTKPlotter.cpp",
                   "plot object",
                   "A mesh must be supplied when plotting an expression");
    }

    // Any type not listed above
    dolfin_error("VTKPlotter.cpp",
                 "plot object",
                 "Object type not supported for plotting");

    return NULL; // not reached
  }
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(std::shared_ptr<const Variable> obj)
  : _initialized(false),
    _plottable(CreateVTKPlottable(obj)),
    vtk_pipeline(new VTKWindowOutputStage()),
    _frame_counter(0),
    _key(to_key(*obj))
{
  parameters = default_parameters();
  _plottable->modify_default_parameters(parameters);
  set_title_from(*obj);
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(std::shared_ptr<const Expression> expression,
                       std::shared_ptr<const Mesh> mesh)
  : _initialized(false),
    _plottable(CreateVTKPlottable(expression, mesh)),
    vtk_pipeline(new VTKWindowOutputStage()),
    _frame_counter(0),
    _key(to_key(*expression))
{
  parameters = default_parameters();
  _plottable->modify_default_parameters(parameters);
  set_title_from(*expression);
}
//----------------------------------------------------------------------------
VTKPlotter::~VTKPlotter()
{
  active_plotters->remove(this);
}
//----------------------------------------------------------------------------
void VTKPlotter::plot(std::shared_ptr<const Variable> variable)
{
  init();

  // Abort if DOLFIN_NOPLOT is set to a nonzero value.
  if (no_plot)
  {
    warning("Environment variable DOLFIN_NOPLOT set: Plotting disabled.");
    return;
  }

  update_pipeline(variable);

  vtk_pipeline->render();

  _frame_counter++;

  // Synthesize key presses from parameters
  Parameter& param_keys = parameters["input_keys"];
  if (param_keys.is_set())
  {
    std::string keys = param_keys;
    for (std::size_t i = 0; i < keys.size(); i++)
    {
      const char c = tolower(keys[i]);
      const int modifiers = (c == keys[i] ? 0 : SHIFT);
      key_pressed(modifiers, c, std::string(&c, 1));
    }
    param_keys.reset();
  }

  if (parameters["interactive"])
    interactive();
}
//----------------------------------------------------------------------------
void VTKPlotter::interactive(bool enter_eventloop)
{
  init();

  // Abort if DOLFIN_NOPLOT is set to a nonzero value, or if 'Q' has
  // been pressed.
  if (no_plot || run_to_end)
    return;

  if (parameters["helptext"])
    vtk_pipeline->set_helptext(get_helptext());

  vtk_pipeline->start_interaction(enter_eventloop);
}
//----------------------------------------------------------------------------
void VTKPlotter::init()
{
  if (_initialized)
  {
    return;
  }
  _initialized = true;

  // Check if environment variable DOLFIN_NOPLOT is set to a nonzero
  // value
  {
    const char *noplot_env;
    noplot_env = getenv("DOLFIN_NOPLOT");
    no_plot = (noplot_env != NULL && strcmp(noplot_env, "0") != 0
               && strcmp(noplot_env, "") != 0);
  }

  // Check if we have a (potential) connection to the X server. In the
  // future, we may instead use a non-gui output stage in this case.
#if defined(Q_WS_X11) || defined(VTK_USE_X)
  if (!getenv("DISPLAY") || strcmp(getenv("DISPLAY"), "") == 0)
  {
    warning("DISPLAY not set, disabling plotting");
    no_plot = true;
  }
#endif

  // Add plotter to pool
  active_plotters->push_back(this);

  // Add a local shared_ptr to the pool. See comment in VTKPlotter.h
  active_plotters_local_copy = active_plotters;
  log(TRACE, "Size of plotter pool is %d.", active_plotters->size());

  // Don't initialize pipeline if no_plot is set, since the pipeline
  // requires a connection to the X server.
  if (no_plot)
    return;

  // Let the plottable set default parameters that depend on user
  // parameters (like scalar-warped 2d plots, which should be elevated
  // only if user doesn't set "mode=off").
  _plottable->modify_user_parameters(parameters);

  // We first initialize the part of the pipeline that the plotter
  // controls.  This is the part from the Poly data mapper and out,
  // including actor, renderer, renderwindow and interaction. It also
  // takes care of the scalar bar and other decorations.

  dolfin_assert(vtk_pipeline);
  vtk_pipeline->init(this, parameters);

  if (parameters["tile_windows"])
  {
    // Adjust window position to not completely overlap previous
    // plots
    std::size_t num_old_plots = active_plotters->size()-1;

    int row=0, col=0, width=0, height=0;
    if (num_old_plots > 0)
    {
      // Get the size of a window that's already decorated, otherwise
      // the frame size may be not include all decoration (on X)
      (*active_plotters->begin())->vtk_pipeline->get_window_size(width, height);

      int swidth, sheight;
      vtk_pipeline->get_screen_size(swidth, sheight);

      // Tile windows horizontally across screen
      int num_rows = swidth/width;
      int num_cols = sheight/height;
      row = num_old_plots % num_rows;
      col = (num_old_plots / num_rows) % num_cols;
    }
    vtk_pipeline->place_window(row*width, col*height);
  }

  // Let the plottable initialize its part of the pipeline
  _plottable->init_pipeline(parameters);
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
  title << "Plot of \"" << variable.name() << "\"" << " ("
        << variable.label() << ")";
  parameters["title"] =  title.str();
}
//----------------------------------------------------------------------------
std::string VTKPlotter::get_helptext()
{
  std::stringstream text;

  text << "Mouse control:\n";
  text << "   Left button: Rotate figure\n";
  text << "   Right button (or scroolwheel): Zoom \n";
  text << "   Middle button (or left+right): Translate figure\n";
  text << "\n";
  text << "Keyboard control:\n";
  text << "   r: Reset zoom\n";
  text << "   f: Fly to the point currently under the mouse pointer\n";
  text << "   s: Synchronize cameras (keep pressed for continuous sync)\n";
  text << "   m: Toggle mesh overlay\n";
  text << "   b: Toggle bounding box\n";
  if (_plottable->dim() <= 2)
    text << "  cv: Toggle cell or vertex indices\n";
  text << "   w: Toggle between wireframe/point/surface view\n";
  text << "  +-: Resize widths (points and lines)\n";
  text << "C-+-: Rescale plot (glyphs and warping)\n";
#ifdef VTK_USE_GL2PS
  text << "  pP: Save plot to png or pdf file\n";
#else
  text << "   p: Save plot to png file\n";
#endif
  text << "  hH: Show help, or print help to console\n";
  text << "  qQ: Continue, or continue to end\n";
  text << " C-c: Abort execution\n";
  text << "\n";

  return text.str();
}
//----------------------------------------------------------------------------
bool VTKPlotter::key_pressed(int modifiers, char key, std::string keysym)
{
  switch (modifiers + key)
  {
  case '+':
    vtk_pipeline->scale_points_lines(1.2);
    vtk_pipeline->render();
    return true;
  case '-':
    vtk_pipeline->scale_points_lines(1.0/1.2);
    vtk_pipeline->render();
    return true;

  case CONTROL + '+': // Up-scale glyphs, etc.
    parameters["scale"] = (double)parameters["scale"] * 1.2;
    rescale();
    vtk_pipeline->render();
    return true;
  case CONTROL + '-': // Down-scale glyphs, etc.
    parameters["scale"] = (double)parameters["scale"] / 1.2;
    rescale();
    vtk_pipeline->render();
    return true;

  case 'b': // Toggle bounding box
    vtk_pipeline->toggle_boundingbox();
    vtk_pipeline->render();
    return true;

  case 'h': // Show help overlay
    vtk_pipeline->toggle_helptext(get_helptext());
    vtk_pipeline->render();
    return true;

  case SHIFT + 'h': // Print helptext to console
    std::cout << get_helptext();
    return true;

  case 'p': // Save plot to file
    write_png();
    return true;

  case SHIFT + 'p': // Save plot to PDF
    write_pdf();
    return true;

  case 'm': // Toggle (secondary) mesh
    {
      vtkSmartPointer<vtkActor> mesh_actor = _plottable->get_mesh_actor();
      bool added = vtk_pipeline->add_viewprop(mesh_actor);
      if (!added)
      {
        mesh_actor->SetVisibility(!mesh_actor->GetVisibility());
      }
      vtk_pipeline->render();
      return true;
    }

  case 'v': // Toggle vertex labels
    {
      if (_plottable->dim() > 2)
        return false;

      // Check if label actor is present. If not get from plottable.
      vtkSmartPointer<vtkActor2D> labels
        = _plottable->get_vertex_label_actor(vtk_pipeline->get_renderer());

      bool added = vtk_pipeline->add_viewprop(labels);
      if (!added)
      {
        // Turn on or off dependent on present state
        labels->SetVisibility(!labels->GetVisibility());
      }

      vtk_pipeline->render();
      return true;
    }

  case 'c': // Toggle cell labels
    {
      if (_plottable->dim() > 2)
        return false;

      // Check if label actor is present. If not get from
      // plottable. If it is, toggle off
      vtkSmartPointer<vtkActor2D> labels
        = _plottable->get_cell_label_actor(vtk_pipeline->get_renderer());

      bool added = vtk_pipeline->add_viewprop(labels);
      if (!added)
      {
        // Turn on or off dependent on present state
        labels->SetVisibility(!labels->GetVisibility());
      }

      vtk_pipeline->render();
      return true;
    }

  case 'w':
    {
      vtk_pipeline->cycle_representation();
      vtk_pipeline->render();
      return true;
    }

  case 's':
  case CONTROL + 's':
  case SHIFT + 's':
  case CONTROL + SHIFT + 's':
    // shift/control may be mouse-interaction modifiers
    {
#if (VTK_MAJOR_VERSION == 6) || ((VTK_MAJOR_VERSION == 5) && (VTK_MINOR_VERSION >= 6))
      vtkCamera* camera = vtk_pipeline->get_camera();
      foreach (VTKPlotter *other, *active_plotters)
      {
        if (other != this)
        {
          other->vtk_pipeline->get_camera()->DeepCopy(camera);
          other->vtk_pipeline->reset_camera_clipping_range();
          other->vtk_pipeline->render();
        }
      }
#else
      warning("Camera sync requires VTK >= 5.6");
#endif
      return true;
    }

  case 'r':
    vtk_pipeline->reset_camera();
    vtk_pipeline->render();
    return true;

  case CONTROL + 'w':
    active_plotters->remove(this);
    return true;

  case CONTROL + 'q':
    active_plotters->clear();
    vtk_pipeline->stop_interaction();
    return true;

  case SHIFT + 'q':
    run_to_end = true;
    vtk_pipeline->stop_interaction();
    return true;

  case CONTROL + 'c':
    dolfin_error("VTKPlotter", "continue execution", "Aborted by user");

  case 'q':
    vtk_pipeline->stop_interaction();
    return true;
  }

  // Not handled
  return false;
}
//----------------------------------------------------------------------------
void VTKPlotter::write_png(std::string filename)
{
  if (no_plot)
    return;

  if (filename.empty())
  {
    // We construct a filename from the given prefix and static
    // counter.  If a file with that filename exists, the counter is
    // incremented until a unique filename is found.
    do
    {
      std::stringstream filenamebuilder;
      filenamebuilder << std::string(parameters["prefix"]);
      filenamebuilder << hardcopy_counter++;
      filename = filenamebuilder.str();
    }
    while (boost::filesystem::exists(filename + ".png"));
  }

  info("Saving plot to file: %s.png", filename.c_str());

  update_pipeline();
  vtk_pipeline->write_png(filename);
}
//----------------------------------------------------------------------------
void VTKPlotter::write_pdf(std::string filename)
{
  if (no_plot)
    return;

  if (filename.empty())
  {
    // We construct a filename from the given prefix and static
    // counter.  If a file with that filename exists, the counter is
    // incremented until a unique filename is found.
    do
    {
      std::stringstream filenamebuilder;
      filenamebuilder << std::string(parameters["prefix"]);
      filenamebuilder << hardcopy_counter++;
      filename = filenamebuilder.str();
    }
    while (boost::filesystem::exists(filename + ".pdf"));
  }

  info("Saving plot to file: %s.pdf", filename.c_str());

  update_pipeline();
  vtk_pipeline->write_pdf(filename);
}
//----------------------------------------------------------------------------
void VTKPlotter::azimuth(double angle)
{
  vtk_pipeline->get_camera()->Azimuth(angle);
}
//----------------------------------------------------------------------------
void VTKPlotter::elevate(double angle)
{
  vtk_pipeline->get_camera()->Elevation(angle);
  vtk_pipeline->reset_camera_clipping_range();
}
//----------------------------------------------------------------------------
void VTKPlotter::dolly(double value)
{
  vtk_pipeline->get_camera()->Dolly(value);
  vtk_pipeline->reset_camera_clipping_range();
}
//----------------------------------------------------------------------------
void VTKPlotter::zoom(double zoomfactor)
{
  vtk_pipeline->get_camera()->Zoom(zoomfactor);
  vtk_pipeline->reset_camera_clipping_range();
}
//----------------------------------------------------------------------------
void VTKPlotter::set_viewangle(double angle)
{
  vtk_pipeline->get_camera()->SetViewAngle(angle);
  vtk_pipeline->reset_camera_clipping_range();
}
//----------------------------------------------------------------------------
void VTKPlotter::set_min_max(double min, double max)
{
  parameters["range_min"] = min;
  parameters["range_max"] = max;
}
//----------------------------------------------------------------------------
void VTKPlotter::add_polygon(const Array<double>& points)
{
  if (no_plot)
    return;

  const std::size_t dim = _plottable->dim();

  if (points.size() % dim != 0)
  {
    warning("VTKPlotter::add_polygon() : Size of array is not a multiple of %d",
            dim);
  }

  const std::size_t numpoints = points.size()/dim;

  vtkSmartPointer<vtkPoints> vtk_points = vtkSmartPointer<vtkPoints>::New();
  vtk_points->SetNumberOfPoints(numpoints);

  double point[3];
  point[2] = 0.0;

  for (std::size_t i = 0; i < numpoints; i++)
  {
    for (std::size_t j = 0; j < dim; j++)
      point[j] = points[i*dim + j];

    vtk_points->InsertPoint(i, point);
  }

  vtkSmartPointer<vtkPolyLine> line = vtkSmartPointer<vtkPolyLine>::New();
  line->GetPointIds()->SetNumberOfIds(numpoints);

  for (std::size_t i = 0; i < numpoints; i++)
    line->GetPointIds()->SetId(i, i);

  vtkSmartPointer<vtkUnstructuredGrid> grid
    = vtkSmartPointer<vtkUnstructuredGrid>::New();
  grid->Allocate(1, 1);

  grid->InsertNextCell(line->GetCellType(), line->GetPointIds());
  grid->SetPoints(vtk_points);

  vtkSmartPointer<vtkGeometryFilter> extract
    = vtkSmartPointer<vtkGeometryFilter>::New();
  #if VTK_MAJOR_VERSION <= 5
  extract->SetInput(grid);
  #else
  extract->SetInputData(grid);
  #endif

  vtkSmartPointer<vtkPolyDataMapper> mapper
    = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(extract->GetOutputPort());

  vtkSmartPointer<vtkActor> polygon_actor = vtkSmartPointer<vtkActor>::New();
  polygon_actor->SetMapper(mapper);

  mapper->SetInputConnection(extract->GetOutputPort());

  polygon_actor->GetProperty()->SetColor(0, 0, 1);
  polygon_actor->GetProperty()->SetLineWidth(1);

  vtk_pipeline->add_viewprop(polygon_actor);
}
//----------------------------------------------------------------------------
void VTKPlotter::rescale()
{
  double range[2];

  const Parameter &range_min = parameters["range_min"];
  const Parameter &range_max = parameters["range_max"];

  if (!range_min.is_set() || !range_max.is_set())
  {
    _plottable->update_range(range);

    // Round small values (< 5% of range) to zero
    const double diff = range[1]-range[0];
    if (diff != 0 && std::abs(range[0]/diff) < 0.05)
      range[0] = 0;
    else if (diff != 0 && std::abs(range[1]/diff) < 0.05)
      range[1] = 0;

    // Round endpoints to 2 significant digits (away from center)
    round_significant_digits(range[0], std::floor, 2);
    round_significant_digits(range[1], std::ceil,  2);
  }

  if (range_min.is_set())
    range[0] = range_min;
  if (range_max.is_set())
    range[1] = range_max;

  _plottable->rescale(range, parameters);
  vtk_pipeline->set_scalar_range(range);

  // The rescale may have changed the scene (scalar/vector warping)
  vtk_pipeline->reset_camera_clipping_range();
}
//----------------------------------------------------------------------------
void VTKPlotter::update_pipeline(std::shared_ptr<const Variable> variable)
{
  if (!is_compatible(variable))
  {
    dolfin_error("VTKPlotter.cpp",
                 "plot object",
                 "The plottable is not compatible with the data");
  }

  Timer timer("VTK update");

  // Process some parameters

  Parameter &wireframe = parameters["wireframe"];
  if (wireframe.is_set())
  {
    vtk_pipeline->cycle_representation(wireframe ? VTK_WIREFRAME : VTK_SURFACE);
    wireframe.reset();
  }

  Parameter &elevation = parameters["elevate"];
  if (elevation.is_set())
  {
    elevate(elevation);
    elevation.reset();
  }

  vtk_pipeline->set_window_title(parameters["title"]);

  // Update the plottable data
  _plottable->update(variable, parameters, _frame_counter);

  // If this is the first render of this plot and/or the rescale
  // parameter is set, we read get the min/max values of the data and
  // process them
  if (_frame_counter == 0 || parameters["rescale"])
    rescale();

  // Set the mapper's connection on each plot. This must be done since
  // the visualization parameters may have changed since the last
  // frame, and the input may hence also have changed

  _plottable->connect_to_output(*vtk_pipeline);
  if (_frame_counter == 0)
    vtk_pipeline->reset_camera();
}
//----------------------------------------------------------------------------
bool VTKPlotter::is_compatible(std::shared_ptr<const Variable> variable) const
{
  return (no_plot || !variable || _plottable->is_compatible(*variable));
}
//----------------------------------------------------------------------------
void VTKPlotter::all_interactive(bool really)
{
  if (active_plotters->size() == 0)
  {
    warning("No plots have been shown yet. Ignoring call to interactive().");
    return;
  }

  if (really)
    run_to_end = false;

  // Prepare interactiveness on every plotter but the first
  foreach (VTKPlotter *plotter, *active_plotters)
  {
    if (plotter != *active_plotters->begin())
      plotter->interactive(false);
  }

  // Start the (global) event loop on the first plotter
  (*active_plotters->begin())->interactive(true);
}


#else

//----------------------------------------------------------------------------
// Implement dummy version of class VTKPlotter even if VTK is not present.
//----------------------------------------------------------------------------

using namespace dolfin;

namespace dolfin
{
  class VTKWindowOutputStage {}; // dummy class
}

VTKPlotter::VTKPlotter(std::shared_ptr<const Variable>)
  : _initialized(false), _frame_counter(0), no_plot(false)
{
  init();
}

VTKPlotter::VTKPlotter(std::shared_ptr<const Expression>,
		       std::shared_ptr<const Mesh>)
{
  init();
}

VTKPlotter::~VTKPlotter() {}

// (Ab)use init() to issue a warning.  We also need to initialize the
// parameter set to avoid tons of warning when running the tests
// without VTK.

void VTKPlotter::init()
{
  parameters = default_parameters();
  warning("Plotting not available. DOLFIN has been compiled without VTK support.");
}

void VTKPlotter::plot         (std::shared_ptr<const Variable>) {}
void VTKPlotter::interactive  (bool)                              {}
void VTKPlotter::write_png    (std::string)                       {}
void VTKPlotter::write_pdf    (std::string)                       {}
void VTKPlotter::azimuth      (double)                            {}
void VTKPlotter::elevate      (double)                            {}
void VTKPlotter::dolly        (double)                            {}
void VTKPlotter::zoom         (double)                            {}
void VTKPlotter::set_viewangle(double)                            {}
void VTKPlotter::set_min_max  (double, double)                    {}
void VTKPlotter::add_polygon  (const Array<double>&)              {}

void VTKPlotter::all_interactive(bool)                            {}
void VTKPlotter::set_key(std::string key)                         {}

bool VTKPlotter::key_pressed(int, char, std::string)
{ return false; }
bool VTKPlotter::is_compatible(std::shared_ptr<const Variable>) const
{ return false; }

std::string        VTKPlotter::to_key(const Variable &) { return ""; }
const std::string& VTKPlotter::key() const              { return _key; }

#endif // HAS_VTK

// Define the static members
std::shared_ptr<std::list<VTKPlotter*>>
VTKPlotter::active_plotters(new std::list<VTKPlotter*>());

int VTKPlotter::hardcopy_counter = 0;
bool VTKPlotter::run_to_end = false;

//----------------------------------------------------------------------------
