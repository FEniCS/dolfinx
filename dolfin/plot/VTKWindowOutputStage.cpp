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
// Last changed: 2012-09-13

#ifdef HAS_VTK

#ifdef HAS_QVTK
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

#ifdef VTK_USE_GL2PS
#include <vtkGL2PSExporter.h>
#endif

#include <boost/filesystem.hpp>

#include <dolfin/common/Timer.h>
#include "VTKWindowOutputStage.h"
#include "VTKPlotter.h"
#include "GenericVTKPlottable.h"

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
  vtkStandardNewMacro(PrivateVTKInteractorStyle)
  //----------------------------------------------------------------------------
#ifdef HAS_QVTK
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
VTKWindowOutputStage::~VTKWindowOutputStage()
{
  // Note: VTK (current 5.6.1) seems to very picky about the order of
  // destruction. This destructor tries to impose an order on the most
  // important stuff.

  std::cout << "Pipeline destroyed\n";

#ifdef HAS_QVTK
  widget.reset(NULL);
#endif

  helptextActor = NULL;
  balloonRep = NULL;
  balloonwidget = NULL;

  _renderer = NULL;
  _renderWindow = NULL;
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

#ifdef HAS_QVTK
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
#ifdef HAS_QVTK
  widget->setWindowTitle(title.c_str());
#else
  _renderWindow->SetWindowName(title.c_str());
#endif
}
//----------------------------------------------------------------------------
std::string VTKWindowOutputStage::get_window_title()
{
#ifdef HAS_QVTK
  return widget->windowTitle().toStdString();
#else
  return _renderWindow->GetWindowName();
#endif
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::close_window()
{
#ifdef HAS_QVTK
  widget->close();
#else
  warning("Window close not implemented on VTK event loop");
#endif
    }
//----------------------------------------------------------------------------
bool VTKWindowOutputStage::resurrect_window()
{
#ifdef HAS_QVTK
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
#ifdef HAS_QVTK
    qApp->exec();
#else
    get_interactor()->Start();
#endif
  }
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::stop_interaction()
{
#ifdef HAS_QVTK
  qApp->quit();
#else
  get_interactor()->TerminateApp();
#endif
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::write_png(std::string filename)
{
  const bool help_visible = helptextActor->GetVisibility();
  helptextActor->VisibilityOff();

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

  if (help_visible)
  {
    helptextActor->VisibilityOn();
    render();
  }
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::write_pdf(std::string filename)
{
#ifdef VTK_USE_GL2PS
  vtkSmartPointer<vtkGL2PSExporter> exporter =
    vtkSmartPointer<vtkGL2PSExporter>::New();
  exporter->SetFilePrefix(filename.c_str());
  //exporter->SetTitle(get_window_title().c_str());
  if (_input == _depthSort)
  {
    // Handle translucency by rasterisation. Commented out because it fails
    // (for me) with GLXBadContextTag error.
    //exporter->Write3DPropsAsRasterImageOn();
  }
  exporter->SetFileFormatToPDF();
  exporter->SetSortToBSP();
  exporter->DrawBackgroundOff();
  exporter->LandscapeOn();
  //exporter->SilentOn();
  exporter->SetRenderWindow(_renderWindow);
  exporter->Write();
#else
  warning("VTK not configured for PDF output");
#endif
}
//----------------------------------------------------------------------------
vtkCamera* VTKWindowOutputStage::get_camera()
{
  return _renderer->GetActiveCamera();
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::reset_camera()
{
  // vtkAxesActor messes up the bounding box, disable it while setting camera
  _axesActor->SetVisibility(false);
  _renderer->ResetCamera();
  _axesActor->SetVisibility(true);
  // but don't clip it
  reset_camera_clipping_range();
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::reset_camera_clipping_range()
{
  _renderer->ResetCameraClippingRange();
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::set_scalar_range(double *range)
{
  _mapper->SetScalarRange(range);
  // Not required, the mapper controls the range.
  //_lut->SetRange(range);
  //_lut->Build();
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
#ifdef HAS_QVTK
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
#ifdef HAS_QVTK
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
#ifdef HAS_QVTK
  widget->move(x, y);
  widget->show();
#else
  _renderWindow->SetPosition(x, y);
#endif
}
//----------------------------------------------------------------------------
bool VTKWindowOutputStage::add_viewprop(vtkSmartPointer<vtkProp> prop)
{
  if (!_renderer->HasViewProp(prop))
  {
    _renderer->AddViewProp(prop);
    return true;
  }
  return false;
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::set_input(vtkSmartPointer<vtkAlgorithmOutput> output)
{
  _input->SetInputConnection(output);
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::set_translucent(bool onoff, uint topo_dim, uint geom_dim)
{
  // In 3D, any translucency in the lut makes the visibility test
  // for cell/vertex labels ineffective.
  // The depth sorting is slow, particularly for glyphs.
  // Hence, set these only when required.

#if (VTK_MAJOR_VERSION == 5) && (VTK_MINOR_VERSION >= 8)
  _lut->SetNanColor(0.0, 0.0, 0.0, (onoff ? 0.05 : 1.0));
#endif

  if (onoff && topo_dim >= 2 && geom_dim == 3)
  {
    _mapper->SetInputConnection(_depthSort->GetOutputPort());
    _input = _depthSort;
  }
  else
  {
    _input = _mapper;
  }
}
//----------------------------------------------------------------------------

#endif // HAS_VTK
