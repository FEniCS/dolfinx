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
// Split from VTKPlotter.h, Joachim B Haga, 2012-09-10
//
// First added:  2012-09-10
// Last changed: 2012-09-10

#ifdef HAS_VTK

#ifdef HAS_QT4
#include <QApplication>
#include <QDesktopWidget>
#include <QVTKWidget.h>
#endif

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
#include <vtkAxesActor.h>
#include <vtkCaptionActor2D.h>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include <dolfin/common/Timer.h>
#include "VTKWindowOutputStage.h"
#include "VTKPlotter.h"
#include "GenericVTKPlottable.h"

#ifdef foreach
#undef foreach
#endif
#define foreach BOOST_FOREACH

using namespace dolfin;

namespace // anonymous
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
      const char key = Interactor->GetKeyCode();
      if (key || !_plotter->keypressCallback())
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
#ifdef HAS_QT4
  void create_qApp()
  {
    if (!qApp)
    {
      static int dummy_argc = 0;
      static char dummy_argv0 = '\0';
      static char *dummy_argv0_ptr = &dummy_argv0;
      new QApplication(dummy_argc, &dummy_argv0_ptr);
    }
  }
#endif
  //----------------------------------------------------------------------------
}
//----------------------------------------------------------------------------
// Class VTKWindowOutputStage
//----------------------------------------------------------------------------
VTKWindowOutputStage::VTKWindowOutputStage()
{
  vtkMapper::GlobalImmediateModeRenderingOn();

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

  _axesActor = vtkSmartPointer<vtkAxesActor>::New();
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::init(VTKPlotter *parent, const Parameters &parameters)
{
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
#else
  _renderWindow->SetInteractor(vtkSmartPointer<vtkRenderWindowInteractor>::New());
  _renderWindow->SetSize(parameters["window_width"], parameters["window_height"]);
#endif
  _renderWindow->GetInteractor()->SetInteractorStyle(style);
  style->SetCurrentRenderer(_renderer);

  // Set some properties that affect the look of things
  _renderer->SetBackground(1, 1, 1);
  _actor->GetProperty()->SetColor(0, 0, 1); //Only used for meshes
  _actor->GetProperty()->SetPointSize(4);   // should be parameter?

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

  if (parameters["axes"])
  {
    //axes->SetShaftTypeToCylinder();
    _axesActor->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(12);
    _axesActor->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(12);
    _axesActor->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(12);
    _axesActor->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->BoldOff();
    _axesActor->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->BoldOff();
    _axesActor->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->BoldOff();
    _axesActor->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->SetColor(0.0, 0.0, 0.0);
    _axesActor->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->SetColor(0.0, 0.0, 0.0);
    _axesActor->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->SetColor(0.0, 0.0, 0.0);
    _axesActor->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->ShadowOn();
    _axesActor->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->ShadowOn();
    _axesActor->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->ShadowOn();
    _renderer->AddActor(_axesActor);
  }
}
//----------------------------------------------------------------------------
vtkRenderWindowInteractor* VTKWindowOutputStage::get_interactor()
{
  return _renderWindow->GetInteractor();
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkRenderer> VTKWindowOutputStage::get_renderer()
{
  return _renderer;
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::scale_points_lines(double factor)
{
  const double pt_size = _actor->GetProperty()->GetPointSize();
  const double l_width = _actor->GetProperty()->GetLineWidth();
  _actor->GetProperty()->SetPointSize(pt_size*factor);
  _actor->GetProperty()->SetLineWidth(l_width*factor);
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::set_helptext(std::string text)
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
//----------------------------------------------------------------------------
void VTKWindowOutputStage::set_window_title(std::string title)
{
#ifdef HAS_QT4
  widget->setWindowTitle(title.c_str());
#else
  _renderWindow->SetWindowName(title.c_str());
#endif
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::close_window()
{
#ifdef HAS_QT4
  widget->close();
#else
  warning("Window close not implemented on VTK event loop");
#endif
    }
//----------------------------------------------------------------------------
bool VTKWindowOutputStage::resurrect_window()
{
#ifdef HAS_QT4
  if (widget->isHidden())
  {
    widget->show();
    return true;
  }
#endif
  return false;
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::start_interaction(bool enter_eventloop)
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
//----------------------------------------------------------------------------
void VTKWindowOutputStage::stop_interaction()
{
#ifdef HAS_QT4
  qApp->quit();
#else
  get_interactor()->TerminateApp();
#endif
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::write_png(std::string filename)
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
//----------------------------------------------------------------------------
vtkCamera* VTKWindowOutputStage::get_camera()
{
  return _renderer->GetActiveCamera();
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::set_scalar_range(double *range)
{
  _mapper->SetScalarRange(range);
  // Not required, the mapper controls the range.
  _lut->SetRange(range);
  _lut->Build();
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::cycle_representation(int new_rep)
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
//----------------------------------------------------------------------------
void VTKWindowOutputStage::render()
{
  Timer timer("VTK render");
  _renderWindow->Render();
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::get_window_size(int& width, int& height)
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
void VTKWindowOutputStage::get_screen_size(int& width, int& height)
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
void VTKWindowOutputStage::place_window(int x, int y)
{
#ifdef HAS_QT4
  widget->move(x, y);
  widget->show();
#else
  _renderWindow->SetPosition(x, y);
#endif
}
//----------------------------------------------------------------------------
bool VTKWindowOutputStage::add_actor(vtkSmartPointer<vtkActor> actor)
{
  if (!_renderer->HasViewProp(actor))
  {
    _renderer->AddActor(actor);
    return true;
  }
  return false;
}
//----------------------------------------------------------------------------
bool VTKWindowOutputStage::add_actor(vtkSmartPointer<vtkActor2D> actor)
{
  if (!_renderer->HasViewProp(actor))
  {
    _renderer->AddActor2D(actor);
    return true;
  }
  return false;
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::set_input(const GenericVTKPlottable &plottable, bool reset_camera)
{
  const bool depthsort = plottable.requires_depthsort();
  if (plottable.dim() < 3 || depthsort)
  {
    // In 3D, only set this when necessary. It makes the visibility test
    // for cell/vertex labels ineffective.
    _lut->SetNanColor(0.0, 0.0, 0.0, 0.05);
  }
  if (depthsort)
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
    // vtkAxesActor messes up the bounding box, disable it while setting camera
    _axesActor->SetVisibility(false);
    _renderer->ResetCamera();
    _axesActor->SetVisibility(true);
  }
}
//----------------------------------------------------------------------------
VTKWindowOutputStage::~VTKWindowOutputStage()
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
//----------------------------------------------------------------------------

#endif // HAS_VTK
