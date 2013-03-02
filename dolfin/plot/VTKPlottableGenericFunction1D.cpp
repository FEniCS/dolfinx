// Copyright (C) 2012 Joachim B Haga
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
// First added:  2012-09-11
// Last changed: 2012-09-17

#ifdef HAS_VTK

#include <vtkXYPlotActor.h>
#include <vtkUnstructuredGrid.h>
#include <vtkProperty2D.h>
#include <vtkGlyphSource2D.h>
#include <vtkTextProperty.h>

#include <dolfin/common/Timer.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>

#include "ExpressionWrapper.h"
#include "VTKWindowOutputStage.h"
#include "VTKPlottableGenericFunction1D.h"

using namespace dolfin;

//----------------------------------------------------------------------------
VTKPlottableGenericFunction1D::VTKPlottableGenericFunction1D(boost::shared_ptr<const Function> function)
  : VTKPlottableGenericFunction(function),
    _actor(vtkSmartPointer<vtkXYPlotActor>::New())
{
  dolfin_assert(dim() == 1);
  // Do nothing
}
//----------------------------------------------------------------------------
VTKPlottableGenericFunction1D::VTKPlottableGenericFunction1D(boost::shared_ptr<const Expression> expression,
                                                             boost::shared_ptr<const Mesh> mesh)
  : VTKPlottableGenericFunction(expression, mesh),
    _actor(vtkSmartPointer<vtkXYPlotActor>::New())
{
  dolfin_assert(dim() == 1);
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction1D::init_pipeline(const Parameters& p)
{
  VTKPlottableGenericFunction::init_pipeline(p);

  _actor->AddInput(grid());
  _actor->SetXValuesToValue();

  _actor->GetProperty()->SetColor(0.0, 0.0, 0.8);
  _actor->GetProperty()->SetLineWidth(1.5);
  _actor->GetProperty()->SetPointSize(4);
  _actor->PlotPointsOn();
  _actor->SetPlotPoints(0, 1);

  _actor->SetXTitle("x");
  _actor->SetYTitle("u(x)");

  _actor->GetAxisLabelTextProperty()->ShadowOff();
  _actor->GetAxisLabelTextProperty()->ItalicOff();
  _actor->GetAxisLabelTextProperty()->SetColor(0, 0, 0);

  _actor->GetAxisTitleTextProperty()->ShadowOff();
  _actor->GetAxisTitleTextProperty()->ItalicOff();
  _actor->GetAxisTitleTextProperty()->SetColor(0, 0, 0);

  _actor->GetPositionCoordinate ()->SetValue(0, 0, 1);
  _actor->GetPosition2Coordinate()->SetValue(1, 1, 0);
  _actor->SetBorder(30);

  #if (VTK_MAJOR_VERSION == 5) && (VTK_MINOR_VERSION >= 6)
  _actor->SetReferenceYValue(0.0);
  #endif
  _actor->SetAdjustYLabels(false); // Use the ranges set in rescale()
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction1D::connect_to_output(VTKWindowOutputStage& output)
{
  output.add_viewprop(_actor);
}
//----------------------------------------------------------------------------
bool VTKPlottableGenericFunction1D::is_compatible(const Variable &var) const
{
  const GenericFunction *function(dynamic_cast<const Function*>(&var));
  const ExpressionWrapper *wrapper(dynamic_cast<const ExpressionWrapper*>(&var));
  const Mesh *mesh(NULL);

  if (function)
  {
    mesh = static_cast<const Function*>(function)->function_space()->mesh().get();
  }
  else if (wrapper)
  {
    mesh = wrapper->mesh().get();
  }

  if (!mesh || mesh->topology().dim() != 1)
  {
    return false;
  }

  return VTKPlottableGenericFunction::is_compatible(var);
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction1D::update(boost::shared_ptr<const Variable> var,
                                           const Parameters& p, int framecounter)
{
  VTKPlottableGenericFunction::update(var, p, framecounter);
  dolfin_assert(dim() == 1);

  double* bounds = grid()->GetBounds(); // [xmin xmax ymin ymax zmin zmax]
  _actor->SetXRange(bounds);
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction1D::rescale(double range[2],
                                            const Parameters& p)
{
  _actor->SetYRange(range);
  #if (VTK_MAJOR_VERSION == 5) && (VTK_MINOR_VERSION >= 6)
  if (range[0] < 0 && range[1] > 0)
    _actor->ShowReferenceYLineOn();
  else
    _actor->ShowReferenceYLineOff();
  #endif
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkActor2D> VTKPlottableGenericFunction1D::get_vertex_label_actor(vtkSmartPointer<vtkRenderer> renderer)
{
  return GenericVTKPlottable::get_vertex_label_actor(renderer);
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkActor2D> VTKPlottableGenericFunction1D::get_cell_label_actor(vtkSmartPointer<vtkRenderer> renderer)
{
  return GenericVTKPlottable::get_cell_label_actor(renderer);
}
//----------------------------------------------------------------------------

#endif
