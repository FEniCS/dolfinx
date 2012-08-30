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
// Last changed: 2012-08-30

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
#include "VTKPlottableDirichletBC.h"
#include "VTKPlotter.h"

#ifdef HAS_QT4

#include <QApplication>
#include <QDesktopWidget>
#include <QVTKWidget.h>

#endif

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
#include <vtkDepthSortPolyData.h>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#ifdef foreach
#undef foreach
#endif
#define foreach BOOST_FOREACH

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

    virtual void OnKeyPress()
    {
      // Only call keypressCallback for non-ascii, to avoid calling twice
      std::string keysym = Interactor->GetKeySym();
      if (keysym.size() == 1 || !_plotter->keypressCallback())
        vtkInteractorStyleTrackballCamera::OnKeyPress();
    }

    virtual void OnChar()
    {
      if (!_plotter->keypressCallback())
        vtkInteractorStyleTrackballCamera::OnChar();
    }

    // A reference to the parent plotter
    VTKPlotter *_plotter;
  };
  vtkStandardNewMacro(PrivateVTKInteractorStyle);
//----------------------------------------------------------------------------
  class PrivateVTKPipeline
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

    // Note: VTK (current 5.6.1) seems to very picky about the order
    // of destruction. It seg faults if the objects are destroyed
    // first (probably before the renderer).
    vtkSmartPointer<vtkTextActor> helptextActor;
    vtkSmartPointer<vtkBalloonRepresentation> balloonRep;
    vtkSmartPointer<vtkBalloonWidget> balloonwidget;

#ifdef HAS_QT4
    boost::scoped_ptr<QVTKWidget> widget;

    static void create_qApp()
    {
      if (!qApp)
      {
        static int dummy_argc = 0;
        static char dummy_argv0 = '\0';
        static char *dummy_argv0_ptr = &dummy_argv0;
        new QApplication(dummy_argc, &dummy_argv0_ptr);
        std::cout << "Created qApp, " << qApp << std::endl;
      }
    }
#endif

  public:

    void init(VTKPlotter *parent, const Parameters &parameters)
    {
      // Initialize objects
      _scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
      _lut = vtkSmartPointer<vtkLookupTable>::New();
      _mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
      _depthSort = vtkSmartPointer<vtkDepthSortPolyData>::New();

      _actor = vtkSmartPointer<vtkActor>::New();
      helptextActor = vtkSmartPointer<vtkTextActor>::New();
      balloonRep = vtkSmartPointer<vtkBalloonRepresentation>::New();
      balloonwidget = vtkSmartPointer<vtkBalloonWidget>::New();

      _renderer = vtkSmartPointer<vtkRenderer>::New();
      _renderWindow = vtkSmartPointer<vtkRenderWindow>::New();

      // Connect the parts
      _mapper->SetLookupTable(_lut);
      _scalarBar->SetLookupTable(_lut);
      _actor->SetMapper(_mapper);
      _renderer->AddActor(_actor);
      _renderWindow->AddRenderer(_renderer);

      // Connect the depth-sort filter to the camera
      _depthSort->SetCamera(_renderer->GetActiveCamera());

      // Set up interactorstyle and connect interactor
      vtkSmartPointer<PrivateVTKInteractorStyle> style =
        vtkSmartPointer<PrivateVTKInteractorStyle>::New();
      style->_plotter = parent;

#ifdef HAS_QT4
      // Set up widget -- make sure a QApplication exists first
      create_qApp();
      widget.reset(new QVTKWidget());
      _renderWindow->SetInteractor(widget->GetInteractor());

      widget->SetRenderWindow(_renderWindow);
      widget->resize(parameters["window_width"], parameters["window_height"]);
      widget->show();
#else
      _renderWindow->SetInteractor(vtkSmartPointer<vtkRenderWindowInteractor>::New());
      _renderWindow->SetSize(parameters["window_width"], parameters["window_height"]);
#endif
      _renderWindow->GetInteractor()->SetInteractorStyle(style);
      style->SetCurrentRenderer(_renderer);

      // Set some properties that affect the look of things
      _renderer->SetBackground(1, 1, 1);
      _lut->SetNanColor(0.0, 0.0, 0.0, 0.05);
      _actor->GetProperty()->SetColor(0, 0, 1); //Only used for meshes
      _actor->GetProperty()->SetPointSize(3);   // should be parameter?
      //_actor->GetProperty()->SetLineWidth(1);

      // Set window stuff
      _scalarBar->SetTextPositionToPrecedeScalarBar();

      // Set the look of scalar bar labels
      vtkSmartPointer<vtkTextProperty> labelprop
	= _scalarBar->GetLabelTextProperty();
      labelprop->SetColor(0, 0, 0);
      labelprop->SetFontSize(20);
      labelprop->ItalicOff();
      labelprop->BoldOff();
      if (parameters["scalarbar"])
        _renderer->AddActor(_scalarBar);
    }

    vtkRenderWindowInteractor* get_interactor()
    {
      return _renderWindow->GetInteractor();
    }

    void set_helptext(std::string text)
    {
      // Add help text actor
      helptextActor->SetPosition(10,10);
      helptextActor->SetInput("Help ");
      helptextActor->GetTextProperty()->SetColor(0.0, 0.0, 0.0);
      helptextActor->GetTextProperty()->SetFontSize(16);
      helptextActor->GetTextProperty()->SetFontFamilyToCourier();
      _renderer->AddActor2D(helptextActor);

      // Set up the representation for the hover-over help text box
      balloonRep->SetOffset(5,5);
      balloonRep->GetTextProperty()->SetFontSize(14);
      balloonRep->GetTextProperty()->BoldOff();
      balloonRep->GetTextProperty()->SetFontFamilyToCourier();
      balloonRep->GetFrameProperty()->SetOpacity(0.7);

      // Set up the actual widget that makes the help text pop up
      balloonwidget->SetInteractor(get_interactor());
      balloonwidget->SetRepresentation(balloonRep);
      balloonwidget->AddBalloon(helptextActor, text.c_str(), NULL);
      balloonwidget->EnabledOn();
    }

    void set_window_title(std::string title)
    {
#ifdef HAS_QT4
      widget->setWindowTitle(title.c_str());
#else
      _renderWindow->SetWindowName(title.c_str());
#endif
    }

    void close_window()
    {
#ifdef HAS_QT4
      widget->close();
#else
      warning("Window close not implemented on VTK event loop");
#endif
    }

    void start_interaction(bool enter_eventloop=true)
    {
      get_interactor()->Initialize();
      render();
      if (enter_eventloop)
      {
#ifdef HAS_QT4
        qApp->exec();
#else
        get_interactor()->Start();
#endif
      }
    }

    void stop_interaction()
    {
#ifdef HAS_QT4
      qApp->quit();
#else
      get_interactor()->TerminateApp();
#endif
    }

    void write_png(std::string filename)
    {
      // FIXME: Remove help-text-actor before hardcopying.

      // Create window to image filter and PNG writer
      vtkSmartPointer<vtkWindowToImageFilter> w2i =
        vtkSmartPointer<vtkWindowToImageFilter>::New();
      vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();

      w2i->SetInput(_renderWindow);
      w2i->Update();
      writer->SetInputConnection(w2i->GetOutputPort());
      writer->SetFileName((filename + ".png").c_str());
      render();
      writer->Modified();
      writer->Write();
    }

    vtkCamera* get_camera()
    {
      return _renderer->GetActiveCamera();
    }

    void set_scalar_range(double *range)
    {
      _mapper->SetScalarRange(range);
      // Not required, the mapper controls the range.
      //_lut->SetRange(range);
      //_lut->Build();
    }

    void cycle_representation(int new_rep=0)
    {
      if (!new_rep)
      {
        const int cur_rep = _actor->GetProperty()->GetRepresentation();
        switch (cur_rep)
        {
        case VTK_SURFACE:   new_rep = VTK_WIREFRAME; break;
        case VTK_WIREFRAME: new_rep = VTK_POINTS;    break;
        default:
        case VTK_POINTS:    new_rep = VTK_SURFACE;   break;
        }
      }
      _actor->GetProperty()->SetRepresentation(new_rep);
    }

    void render()
    {
      Timer timer("VTK render");
      _renderWindow->Render();
    }

    void get_window_size(int& width, int& height)
    {
#ifdef HAS_QT4
      QSize size = widget->frameSize();
      width = size.width();
      height = size.height();
#else
      get_interactor()->GetSize(width, height);
      // Guess window decoration (frame) size
      width += 6;
      height += 30;
#endif
    }
    //----------------------------------------------------------------------------
    void get_screen_size(int& width, int& height)
    {
#ifdef HAS_QT4
      QRect geom = QApplication::desktop()->availableGeometry();
      width = geom.width();
      height = geom.height();
#else
      int *size = _renderWindow->GetScreenSize();
      width = size[0];
      height = size[1];
#endif
    }
    //----------------------------------------------------------------------------
    void set_window_position(int x, int y)
    {
#ifdef HAS_QT4
      widget->move(x, y);
#else
      _renderWindow->SetPosition(x, y);
#endif
    }

    bool add_actor(vtkSmartPointer<vtkProp> actor)
    {
      if (!_renderer->HasViewProp(actor))
      {
        _renderer->AddActor(actor);
        return true;
      }
      return false;
    }

    void set_input(const GenericVTKPlottable &plottable, bool reset_camera)
    {
      if (plottable.requires_depthsort())
      {
        std::cout << "Depth sort\n";
        _depthSort->SetInputConnection(plottable.get_output());
        _mapper->SetInputConnection(_depthSort->GetOutputPort());
      }
      else
      {
        _mapper->SetInputConnection(plottable.get_output());
      }

      if (reset_camera)
      {
        _renderer->ResetCamera();
      }
    }

    ~PrivateVTKPipeline()
    {
      // Note: VTK (current 5.6.1) seems to very picky about the order of
      // destruction. This destructor tries to impose an order on the most
      // important stuff.

      std::cout << "Pipeline destroyed\n";

#ifdef HAS_QT4
      widget.reset(NULL);
#endif

      helptextActor = NULL;
      balloonRep = NULL;
      balloonwidget = NULL;

      _renderer = NULL;
      _renderWindow = NULL;
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
  _key(to_key(*mesh))
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
  _key(to_key(*function))
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
  _key(to_key(*expression))
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
  _key(to_key(*wrapper->expression()))
{
  parameters = default_parameters();
  set_title_from(*wrapper->expression());
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::VTKPlotter(boost::shared_ptr<const DirichletBC> bc) :
  _plottable(boost::shared_ptr<GenericVTKPlottable>(
    new VTKPlottableDirichletBC(bc))),
  vtk_pipeline(new PrivateVTKPipeline()),
  _frame_counter(0),
  _key(to_key(*bc))
{
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
  _key(to_key(*mesh_function))
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
  _key(to_key(*mesh_function))
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
  _key(to_key(*mesh_function))
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
  _key(to_key(*mesh_function))
{
  parameters = default_parameters();
  set_title_from(*mesh_function);
  init();
}
//----------------------------------------------------------------------------
VTKPlotter::~VTKPlotter()
{
  all_plotters->remove(this);
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

  vtk_pipeline->render();

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

  if (parameters["helptext"])
  {
    vtk_pipeline->set_helptext(get_helptext());
  }

  vtk_pipeline->start_interaction(enter_eventloop);
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

  // We first initialize the part of the pipeline that the plotter controls.
  // This is the part from the Poly data mapper and out, including actor,
  // renderer, renderwindow and interaction. It also takes care of the scalar
  // bar and other decorations.

  dolfin_assert(vtk_pipeline);
  vtk_pipeline->init(this, parameters);

  // Adjust window position to not completely overlap previous plots
  dolfin::uint num_old_plots = VTKPlotter::all_plotters->size()-1;

  int width, height;
  vtk_pipeline->get_window_size(width, height);

  int swidth, sheight;
  vtk_pipeline->get_screen_size(swidth, sheight);

  // Tile windows horizontally across screen
  int num_rows = swidth/width;
  int num_cols = sheight/height;
  int row = num_old_plots % num_rows;
  int col = (num_old_plots / num_rows) % num_cols;

  vtk_pipeline->set_window_position(row*width, col*height);

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
  text << "   w: Toggle wireframe/point/surface view\n";
  text << "   f: Fly to the point currently under the mouse pointer\n";
  text << "   s: Synchronize cameras (keep pressed for continuous sync)\n";
  text << "   p: Add bounding box\n";
  text << "   v: Toggle vertex indices on/off\n";
  text << "   h: Save plot to file\n";
  text << "   q: Continue\n";
  text << "\n";
#ifdef HAS_QT4
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
  static const int CONTROL = 0x300;

  const std::string key = vtk_pipeline->get_interactor()->GetKeySym();
  const int modifiers = (SHIFT   * !!vtk_pipeline->get_interactor()->GetShiftKey() +
                         ALT     * !!vtk_pipeline->get_interactor()->GetAltKey()   +
                         CONTROL * !!vtk_pipeline->get_interactor()->GetControlKey());

  std::cout << "Keypress: " << key << '|' << modifiers << std::endl;

  if (key.size() != 1)
  {
    return false;
  }

  switch (modifiers + tolower(key[0]))
  {
  case 'h': // Save plot to file
    write_png();
    return true;

  case 'v': // Toggle vertex labels
    {
      // Check if label actor is present. If not get from plottable. If it
      // is, toggle off
      vtkSmartPointer<vtkActor2D> labels = _plottable->get_vertex_label_actor();

      bool added = vtk_pipeline->add_actor(labels);
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
    {
      vtkCamera* camera = vtk_pipeline->get_camera();
      foreach (VTKPlotter *other, *all_plotters)
      {
        if (other != this)
        {
          other->vtk_pipeline->get_camera()->DeepCopy(camera);
          other->vtk_pipeline->render();
        }
      }
      return true;
    }

  case CONTROL + 'w':
    vtk_pipeline->close_window();
    all_plotters->remove(this);
    return true;

  case CONTROL + 'q':
    foreach (VTKPlotter *plotter, *all_plotters)
    {
      plotter->vtk_pipeline->close_window();
    }
    all_plotters->clear();
    // FALL THROUGH
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
  dolfin_assert(vtk_pipeline);

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
  vtk_pipeline->write_png(filename);
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
}
//----------------------------------------------------------------------------
void VTKPlotter::dolly(double value)
{
  vtk_pipeline->get_camera()->Dolly(value);
}
//----------------------------------------------------------------------------
void VTKPlotter::set_viewangle(double angle)
{
  vtk_pipeline->get_camera()->SetViewAngle(angle);
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

  vtkSmartPointer<vtkActor> polygon_actor = vtkSmartPointer<vtkActor>::New();
  polygon_actor->SetMapper(mapper);

  mapper->SetInputConnection(extract->GetOutputPort());

  polygon_actor->GetProperty()->SetColor(0, 0, 1);
  polygon_actor->GetProperty()->SetLineWidth(1);

  vtk_pipeline->add_actor(polygon_actor);
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

  Timer timer("VTK update");

  // Process some parameters
  if (parameters["wireframe"])
    vtk_pipeline->cycle_representation(VTK_WIREFRAME);

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

    vtk_pipeline->set_scalar_range(range);
  }

  // Set the mapper's connection on each plot. This must be done since the
  // visualization parameters may have changed since the last frame, and
  // the input may hence also have changed
  vtk_pipeline->set_input(*_plottable, _frame_counter==0);
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
    // Prepare interactiveness on every plotter but the first
    foreach (VTKPlotter *plotter, *all_plotters)
    {
      if (plotter != *all_plotters->begin())
        plotter->interactive(false);
    }

    // Start the vtk eventloop on the first plotter
    (*all_plotters->begin())->interactive(true);
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
void VTKPlotter::write_png          (std::string){}
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
