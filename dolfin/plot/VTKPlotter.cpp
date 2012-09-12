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
// Last changed: 2012-09-12

#include <dolfin/common/Array.h>
#include <dolfin/common/Timer.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/generation/CSGGeometry.h>
#include "ExpressionWrapper.h"
#include "VTKPlotter.h"

#ifdef HAS_VTK

#include "VTKWindowOutputStage.h"
#include "VTKPlottableGenericFunction.h"
#include "VTKPlottableMesh.h"
#include "VTKPlottableMeshFunction.h"
#include "VTKPlottableDirichletBC.h"
#include "VTKPlottableCSGGeometry.h"

#ifdef HAS_QVTK
#include <QApplication>
#include <QtGlobal>
#endif

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
template <class T>
VTKPlotter::VTKPlotter(boost::shared_ptr<T> t)
  : _initialized(false),
    _plottable(CreateVTKPlottable(t)),
    vtk_pipeline(new VTKWindowOutputStage()),
    _frame_counter(0),
    _key(to_key(*t))
{
  parameters = default_parameters();
  parameters.update(_plottable->default_parameters());
  set_title_from(*t);
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const Expression> expression,
    boost::shared_ptr<const Mesh> mesh)
  : _initialized(false),
    _plottable(CreateVTKPlottable(expression, mesh)),
    vtk_pipeline(new VTKWindowOutputStage()),
    _frame_counter(0),
    _key(to_key(*expression))
{
  parameters = default_parameters();
  parameters.update(_plottable->default_parameters());
  set_title_from(*expression);
}
//----------------------------------------------------------------------------
VTKPlotter::~VTKPlotter()
{
  active_plotters->remove(this);
}
//----------------------------------------------------------------------------
void VTKPlotter::plot(boost::shared_ptr<const Variable> variable)
{
  init();

  // Abort if DOLFIN_NOPLOT is set to a nonzero value.
  if (no_plot)
  {
    warning("Environment variable DOLFIN_NOPLOT set: Plotting disabled.");
    return;
  }

  if (vtk_pipeline->resurrect_window())
  {
    active_plotters->push_back(this);
  }

  update_pipeline(variable);

  vtk_pipeline->render();

  _frame_counter++;

  if (parameters["interactive"])
  {
    interactive();
  }
#ifdef HAS_QVTK
  else
  {
    qApp->processEvents();
  }
#endif
}
//----------------------------------------------------------------------------
void VTKPlotter::interactive(bool enter_eventloop)
{
  init();

  // Abort if DOLFIN_NOPLOT is set to a nonzero value, or if 'Q' has been pressed.
  if (no_plot || run_to_end)
  {
    return;
  }

  if (parameters["helptext"])
  {
    vtk_pipeline->set_helptext(get_helptext());
  }

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

  // Check if environment variable DOLFIN_NOPLOT is set to a nonzero value
  {
    const char *noplot_env;
    noplot_env = getenv("DOLFIN_NOPLOT");
    no_plot = (noplot_env != NULL && strcmp(noplot_env, "0") != 0 && strcmp(noplot_env, "") != 0);
  }

  // Check if we have a (potential) connection to the X server. In the future,
  // we may instead use a non-gui output stage in this case.
#if defined(Q_WS_X11) || defined(VTK_USE_X) // <QtGlobal>, <vtkToolkits.h>
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

  // Don't initialize pipeline if no_plot is set, since the pipeline requires a
  // connection to the X server.
  if (no_plot)
  {
    return;
  }

  // We first initialize the part of the pipeline that the plotter controls.
  // This is the part from the Poly data mapper and out, including actor,
  // renderer, renderwindow and interaction. It also takes care of the scalar
  // bar and other decorations.

  dolfin_assert(vtk_pipeline);
  vtk_pipeline->init(this, parameters);

  // Adjust window position to not completely overlap previous plots.
  dolfin::uint num_old_plots = active_plotters->size()-1;

  int row=0, col=0, width=0, height=0;
  if (num_old_plots > 0)
  {
    // Get the size of a window that's already decorated, otherwise the frame
    // size may be not include all decoration (on X)
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
  title << "Plot of \"" << variable.name() << "\"" << " (" << variable.label() << ")";
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
  text << "   p: Toggle bounding box\n";
  text << "   v: Toggle vertex indices\n";
  text << "   w: Toggle wireframe/point/surface view\n";
  text << "  +-: Resize points and lines\n";
  text << "   h: Save plot to png (raster) file\n";
#ifdef VTK_USE_GL2PS
  text << "   H: Save plot to pdf (vector) file\n";
#endif
  text << "   q: Continue\n";
  text << "   Q: Continue to end\n";
  text << " C-C: Abort execution\n";
  text << "\n";
#ifdef HAS_QVTK
  text << "Window control:\n";
  text << " C-w: Close plot window\n";
  text << " C-q: Close all plot windows\n";
#endif
  return text.str();
}
//----------------------------------------------------------------------------
bool VTKPlotter::keypressCallback()
{
  static const int SHIFT   = 0x100; // Preserve the low word so a char can be added
  static const int ALT     = 0x200;
  static const int CONTROL = 0x400;

  // Note: ALT key doesn't seem to be usable as a modifier.
  std::string keysym = vtk_pipeline->get_interactor()->GetKeySym();
  char key = vtk_pipeline->get_interactor()->GetKeyCode();
  int modifiers = (SHIFT   * !!vtk_pipeline->get_interactor()->GetShiftKey() +
                   ALT     * !!vtk_pipeline->get_interactor()->GetAltKey()   +
                   CONTROL * !!vtk_pipeline->get_interactor()->GetControlKey());
  if (keysym.size() == 1)
  {
    // Fix for things like shift+control+q which isn't sent correctly
    key = keysym[0];
  }

  key = tolower(key);
  if (key && key == toupper(key))
  {
    // Things like '+', '&' which are not really shifted
    modifiers &= ~SHIFT;
  }

  std::cout << "Keypress: " << key << "|" << modifiers << " (" << keysym << ")\n";

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

  case 'h': // Save plot to file
    write_png();
    return true;

  case SHIFT + 'h': // Save plot to PDF
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
      // Check if label actor is present. If not get from plottable.
      vtkSmartPointer<vtkActor2D> labels = _plottable->get_vertex_label_actor(vtk_pipeline->get_renderer());

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
      // Check if label actor is present. If not get from plottable. If it
      // is, toggle off
      vtkSmartPointer<vtkActor2D> labels = _plottable->get_cell_label_actor(vtk_pipeline->get_renderer());

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
      parameters["wireframe"].reset(); // Don't override in plot()
      return true;
    }

  case 's':
  case CONTROL + 's':
  case SHIFT + 's':
  case CONTROL + SHIFT + 's':
    // shift/control may be mouse-interaction modifiers
    {
#if (VTK_VERSION_MAJOR == 5) && (VTK_VERSION_MINOR >= 6)
      vtkCamera* camera = vtk_pipeline->get_camera();
      foreach (VTKPlotter *other, *active_plotters)
      {
        if (other != this)
        {
          other->vtk_pipeline->get_camera()->DeepCopy(camera);
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
    vtk_pipeline->close_window();
    active_plotters->remove(this);
    return true;

  case CONTROL + 'q':
    foreach (VTKPlotter *plotter, *active_plotters)
    {
      plotter->vtk_pipeline->close_window();
    }
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

  if (filename.empty()) {
    // We construct a filename from the given prefix and static counter.
    // If a file with that filename exists, the counter is incremented
    // until a unique filename is found.
    do {
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

  if (filename.empty()) {
    // We construct a filename from the given prefix and static counter.
    // If a file with that filename exists, the counter is incremented
    // until a unique filename is found.
    do {
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

  vtkSmartPointer<vtkActor> polygon_actor = vtkSmartPointer<vtkActor>::New();
  polygon_actor->SetMapper(mapper);

  mapper->SetInputConnection(extract->GetOutputPort());

  polygon_actor->GetProperty()->SetColor(0, 0, 1);
  polygon_actor->GetProperty()->SetLineWidth(1);

  vtk_pipeline->add_viewprop(polygon_actor);
}
//----------------------------------------------------------------------------
void VTKPlotter::update_pipeline(boost::shared_ptr<const Variable> variable)
{
  if (!is_compatible(variable))
  {
    dolfin_error("VTKPlotter.cpp",
                 "plot()",
                 "The plottable is not compatible with the data");
  }

  Timer timer("VTK update");

  // Process some parameters
  Parameter &wireframe = parameters["wireframe"];
  if (wireframe.is_set())
  {
    vtk_pipeline->cycle_representation(wireframe ? VTK_WIREFRAME : VTK_SURFACE);
  }

  vtk_pipeline->set_window_title(parameters["title"]);

  // Update the plottable data
  _plottable->update(variable, parameters, _frame_counter);

  // If this is the first render of this plot and/or the rescale parameter
  // is set, we read get the min/max values of the data and process them
  if (_frame_counter == 0 || parameters["rescale"])
  {
    double range[2];

    const Parameter &range_min = parameters["range_min"];
    const Parameter &range_max = parameters["range_max"];

    if (!range_min.is_set() || !range_max.is_set())
    {
      _plottable->update_range(range);

      // Round small values (<5% of range) to zero
      const double diff = range[1]-range[0];
      if (diff != 0 && std::abs(range[0]/diff) < 0.05)
        range[0] = 0;
      else if (diff != 0 && std::abs(range[1]/diff) < 0.05)
        range[1] = 0;

      // Round endpoints to 2 significant digits (away from center)
      round_significant_digits(range[0], std::floor, 2);
      round_significant_digits(range[1], std::ceil,  2);
    }

    if (range_min.is_set()) range[0] = range_min;
    if (range_max.is_set()) range[1] = range_max;

    _plottable->rescale(range, parameters);
    vtk_pipeline->set_scalar_range(range);
    // The rescale may have changed the scene (scalar/vector warping)
    vtk_pipeline->reset_camera_clipping_range();
  }

  // Set the mapper's connection on each plot. This must be done since the
  // visualization parameters may have changed since the last frame, and
  // the input may hence also have changed
  _plottable->connect_to_output(*vtk_pipeline);
  if (_frame_counter == 0)
    vtk_pipeline->reset_camera();
}
//----------------------------------------------------------------------------
bool VTKPlotter::is_compatible(boost::shared_ptr<const Variable> variable) const
{
  return (!variable || _plottable->is_compatible(*variable));
}
//----------------------------------------------------------------------------
void VTKPlotter::all_interactive(bool really)
{
  if (active_plotters->size() == 0)
    warning("No plots have been shown yet. Ignoring call to interactive().");
  else
  {
    if (really)
    {
      run_to_end = false;
    }

    // Prepare interactiveness on every plotter but the first
    foreach (VTKPlotter *plotter, *active_plotters)
    {
      if (plotter != *active_plotters->begin())
        plotter->interactive(false);
    }

    // Start the (global) event loop on the first plotter
    (*active_plotters->begin())->interactive(true);
  }
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

template <class T>
VTKPlotter::VTKPlotter(boost::shared_ptr<T> t) { init(); }
VTKPlotter::VTKPlotter(boost::shared_ptr<const Expression> e,
		       boost::shared_ptr<const Mesh> mesh)  { init(); }
VTKPlotter::~VTKPlotter() {}

// (Ab)use init() to issue a warning.
// We also need to initialize the parameter set to avoid tons of warning
// when running the tests without VTK.

void VTKPlotter::init()
{
  parameters = default_parameters();
  warning("Plotting not available. DOLFIN has been compiled without VTK support.");
}
void VTKPlotter::plot         (boost::shared_ptr<const Variable>) {}
void VTKPlotter::interactive  (bool)                              {}
void VTKPlotter::write_png    (std::string)                       {}
void VTKPlotter::azimuth      (double)                            {}
void VTKPlotter::elevate      (double)                            {}
void VTKPlotter::dolly        (double)                            {}
void VTKPlotter::set_viewangle(double)                            {}
void VTKPlotter::set_min_max  (double, double)                    {}
void VTKPlotter::add_polygon  (const Array<double>&)              {}

void VTKPlotter::all_interactive(bool)                            {}
void VTKPlotter::set_key(std::string key)                         {}

bool VTKPlotter::keypressCallback()                                     { return false; }
bool VTKPlotter::is_compatible(boost::shared_ptr<const Variable>) const { return false; }

std::string        VTKPlotter::to_key(const Variable &) { return ""; }
const std::string& VTKPlotter::key() const              { return _key; }

#endif // HAS_VTK

// Define the static members
boost::shared_ptr<std::list<VTKPlotter*> > VTKPlotter::active_plotters(new std::list<VTKPlotter*>());
int VTKPlotter::hardcopy_counter = 0;
bool VTKPlotter::run_to_end = false;

//---------------------------------------------------------------------------
// Instantiate constructors for valid types
//---------------------------------------------------------------------------

// Must instantiate both const and non-const shared_ptr<T>s, no implicit conversion; see
// http://stackoverflow.com/questions/5600150/c-template-instantiation-with-shared-ptr-to-const-t
#define INSTANTIATE(T)                                                  \
  template VTKPlotter::VTKPlotter(boost::shared_ptr<const T >); \
  template VTKPlotter::VTKPlotter(boost::shared_ptr<T >);

INSTANTIATE(CSGGeometry)
INSTANTIATE(DirichletBC)
INSTANTIATE(ExpressionWrapper)
INSTANTIATE(Function)
INSTANTIATE(Mesh)
INSTANTIATE(MeshFunction<bool>)
INSTANTIATE(MeshFunction<double>)
INSTANTIATE(MeshFunction<float>)
INSTANTIATE(MeshFunction<int>)
INSTANTIATE(MeshFunction<uint>)
