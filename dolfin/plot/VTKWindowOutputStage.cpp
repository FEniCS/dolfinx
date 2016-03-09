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
// Modified by Benjamin Kehlet, 2012
//
// First added:  2012-09-10
// Last changed: 2012-11-13

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


// The below is a work-around for Intel compilers which have a problem
// with unnamed namespaces. See
// http://software.intel.com/en-us/articles/compiler-reports-error-1757-when-compiling-chromium-os-code
// and https://bugs.launchpad.net/dolfin/+bug/1086526

namespace d_anonymous { /* empty body */ }
using namespace d_anonymous;

namespace d_anonymous
{
  //----------------------------------------------------------------------------
  class PrivateVTKInteractorStyle : public vtkInteractorStyleTrackballCamera
  {
    // Create a new style instead of observer callbacks, so that we
    // can intercept keypresses (like q/e) reliably.
  public:
    PrivateVTKInteractorStyle() : _plotter(NULL), _highlighted(false) {}

    static PrivateVTKInteractorStyle* New();
    vtkTypeMacro(PrivateVTKInteractorStyle, vtkInteractorStyleTrackballCamera)

    virtual void OnKeyPress()
    {
      // Only call keypressCallback for non-ascii, to avoid calling twice
      const char key = Interactor->GetKeyCode();
      if (key || !handle_keypress())
        vtkInteractorStyleTrackballCamera::OnKeyPress();
    }

    virtual void OnChar()
    {
      if (!handle_keypress())
        vtkInteractorStyleTrackballCamera::OnChar();
    }

    bool handle_keypress()
    {
      // Note: ALT key doesn't seem to be usable as a modifier.
      std::string keysym = Interactor->GetKeySym();
      char key = Interactor->GetKeyCode();
      int modifiers = (static_cast<int>(VTKPlotter::Modifiers::SHIFT) * !!Interactor->GetShiftKey() +
                       static_cast<int>(VTKPlotter::Modifiers::ALT) * !!Interactor->GetAltKey() +
                       static_cast<int>(VTKPlotter::Modifiers::CONTROL) * !!Interactor->GetControlKey());
      if (keysym.size() == 1)
      {
        // Fix for things like shift+control+q which isn't sent correctly
        key = keysym[0];
      }

      key = tolower(key);
      if (key && key == toupper(key))
      {
        // Things like '+', '&' which are not really shifted
        modifiers &= ~static_cast<int>(VTKPlotter::Modifiers::SHIFT);
      }

      log(DBG, "Keypress: %c|%d (%s)", key, modifiers, keysym.c_str());
      return _plotter->key_pressed(modifiers, key, keysym);
    }

    // A reference to the parent plotter
    VTKPlotter *_plotter;

    // A flag to indicate bounding box is visible
    bool _highlighted;

  };
  vtkStandardNewMacro(PrivateVTKInteractorStyle)
  //----------------------------------------------------------------------------
  class PrivateVTKBalloonWidget : public vtkBalloonWidget
  {
  public:

    PrivateVTKBalloonWidget() : _force_visible(false) {}

    static PrivateVTKBalloonWidget* New();
    vtkTypeMacro(PrivateVTKBalloonWidget, vtkBalloonWidget)

    virtual int SubclassEndHoverAction()
    {
      if (_force_visible)
        return 1;
      else
        return vtkBalloonWidget::SubclassEndHoverAction();
    }

    void toggle_popup(std::string text, vtkBalloonRepresentation *rep) {
      // This callmethod is only available from vtk 5.6:
      //vtkBalloonRepresentation *rep = GetBalloonRepresentation();
      double e[2] = {10, 10};

      _force_visible = !_force_visible;
      if (_force_visible)
      {
        EnabledOn();
        rep->SetBalloonText(text.c_str());
        rep->StartWidgetInteraction(e);
      }
      else
        rep->EndWidgetInteraction(e);
    }

    bool _force_visible;

  };
  vtkStandardNewMacro(PrivateVTKBalloonWidget)
  //----------------------------------------------------------------------------
  unsigned char gauss_120[256*4] =
  {
  #include "gauss_120.dat"
  };
}
//----------------------------------------------------------------------------
// Class VTKWindowOutputStage
//----------------------------------------------------------------------------
VTKWindowOutputStage::VTKWindowOutputStage()
{
  vtkMapper::GlobalImmediateModeRenderingOn(); // FIXME: Check if faster or not

  // Initialize objects
  _scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
  _lut = vtkSmartPointer<vtkLookupTable>::New();
  _mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  _depthSort = vtkSmartPointer<vtkDepthSortPolyData>::New();

  _actor = vtkSmartPointer<vtkActor>::New();
  helptextActor = vtkSmartPointer<vtkTextActor>::New();
  balloonRep = vtkSmartPointer<vtkBalloonRepresentation>::New();
  balloonwidget = vtkSmartPointer<PrivateVTKBalloonWidget>::New();

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

  helptextActor = NULL;
  balloonRep = NULL;
  balloonwidget = NULL;

  _renderer = NULL;
  _renderWindow = NULL;
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::init(VTKPlotter *parent, const Parameters& p)
{
  // Connect the parts
  _mapper->SetLookupTable(_lut);
  _scalarBar->SetLookupTable(_lut);
  _actor->SetMapper(_mapper);
  _renderer->AddActor(_actor);
  _renderWindow->AddRenderer(_renderer);

  // Load the lookup table
  vtkSmartPointer<vtkUnsignedCharArray> lut_data
    = vtkSmartPointer<vtkUnsignedCharArray>::New();
  lut_data->SetNumberOfComponents(4);
  lut_data->SetArray(gauss_120, 256*4, 1);
  _lut->SetTable(lut_data.GetPointer());

  // Connect the depth-sort filter to the camera
  _depthSort->SetCamera(_renderer->GetActiveCamera());

  // Set up interactorstyle and connect interactor
  vtkSmartPointer<PrivateVTKInteractorStyle> style =
    vtkSmartPointer<PrivateVTKInteractorStyle>::New();
  style->_plotter = parent;

  _renderWindow->SetInteractor(vtkSmartPointer<vtkRenderWindowInteractor>::New());
  const int width  = p["window_width"];
  const int height = p["window_height"];
  if (width > 0 && height > 0)
    _renderWindow->SetSize(width, height);

  _renderWindow->GetInteractor()->SetInteractorStyle(style);
  style->SetCurrentRenderer(_renderer);

  // Set some properties that affect the look of things
  _renderer->SetBackground(1, 1, 1);
  _actor->GetProperty()->SetColor(0, 0, 1); //Only used for meshes
  _actor->GetProperty()->SetPointSize(4);   // should be parameter?

  // Set window stuff
  _scalarBar->SetTitle(" "); // To avoid uninitialized-warning in VTK 6
  _scalarBar->SetTextPositionToPrecedeScalarBar();

  // Set the look of scalar bar labels
  vtkSmartPointer<vtkTextProperty> labelprop
    = _scalarBar->GetLabelTextProperty();
  labelprop->SetColor(0, 0, 0);
  labelprop->SetFontSize(20);
  labelprop->ItalicOff();
  labelprop->BoldOff();
  if (p["scalarbar"])
    _renderer->AddActor(_scalarBar);

  if (p["axes"])
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

  // Create help text actor
  helptextActor->SetPosition(10, 10);
  helptextActor->SetInput("Help ");
  helptextActor->GetTextProperty()->SetColor(0.0, 0.0, 0.0);
  helptextActor->GetTextProperty()->SetFontSize(16);
  helptextActor->GetTextProperty()->SetFontFamilyToCourier();

  // Set up the representation for the hover-over help text box
  balloonRep->SetOffset(5, 5);
  balloonRep->GetTextProperty()->SetFontSize(14);
  balloonRep->GetTextProperty()->BoldOff();
  balloonRep->GetTextProperty()->SetFontFamilyToCourier();
  balloonRep->GetFrameProperty()->SetOpacity(0.7);

  // Set up the actual widget that makes the help text pop up
  balloonwidget->SetInteractor(get_interactor());
  balloonwidget->SetRepresentation(balloonRep);
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
  _renderer->AddActor2D(helptextActor);

  // Add the balloon text to the actor
  balloonwidget->AddBalloon(helptextActor, text.c_str(), NULL);
  balloonwidget->EnabledOn();
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::set_window_title(std::string title)
{
  _renderWindow->SetWindowName(title.c_str());
}
//----------------------------------------------------------------------------
std::string VTKWindowOutputStage::get_window_title()
{
  return _renderWindow->GetWindowName();
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::start_interaction(bool enter_eventloop)
{
  get_interactor()->Initialize();
  render();
  if (enter_eventloop)
    get_interactor()->Start();
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::stop_interaction()
{
  get_interactor()->TerminateApp();
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
  vtkSmartPointer<vtkGL2PSExporter> exporter
    = vtkSmartPointer<vtkGL2PSExporter>::New();
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
  // Normally the mapper controls the lut range. But if the mapper isn't
  // activated (no visible actors), then the lut update will be delayed.
  _lut->SetRange(range);
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
void VTKWindowOutputStage::toggle_boundingbox()
{
  PrivateVTKInteractorStyle *style
    = dynamic_cast<PrivateVTKInteractorStyle*>(get_interactor()->GetInteractorStyle());
  if (style)
  {
    style->_highlighted = !style->_highlighted;
    style->HighlightProp(style->_highlighted ? _actor : NULL);
  }
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::toggle_helptext(std::string text)
{
  PrivateVTKBalloonWidget *balloon
    = dynamic_cast<PrivateVTKBalloonWidget*>((vtkBalloonWidget*)balloonwidget);
  balloon->toggle_popup(text, balloonRep);
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
  get_interactor()->GetSize(width, height);
  // Guess window decoration (frame) size
  width += 6;
  height += 30;
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::get_screen_size(int& width, int& height)
{
  int *size = _renderWindow->GetScreenSize();
  width = size[0];
  height = size[1];
}
//----------------------------------------------------------------------------
void VTKWindowOutputStage::place_window(int x, int y)
{
  _renderWindow->SetPosition(x, y);
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
void VTKWindowOutputStage::set_translucent(bool onoff, std::size_t topo_dim,
                                           std::size_t geom_dim)
{
  // In 3D, any translucency in the lut makes the visibility test
  // for cell/vertex labels ineffective.
  // The depth sorting is slow, particularly for glyphs.
  // Hence, set these only when required.

  #if (VTK_MAJOR_VERSION == 6) || ((VTK_MAJOR_VERSION == 5) && (VTK_MINOR_VERSION >= 8))
  _lut->SetNanColor(0.0, 0.0, 0.0, (onoff ? 0.05 : 1.0));
  #endif

  if (onoff && topo_dim >= 2 && geom_dim == 3)
  {
    _mapper->SetInputConnection(_depthSort->GetOutputPort());
    _input = _depthSort;
  }
  else
    _input = _mapper;
}
//----------------------------------------------------------------------------

#endif // HAS_VTK
