// Copyright (C) 2012 Joachim Berdal Haga
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
// First added:  2012-09-14
// Last changed: 2012-09-18

#include <vtkNew.h>
#include <vtkCellPicker.h>
#include <vtkRenderer.h>

#include <dolfin/plot/VTKWindowOutputStage.h>

#include "Plotter.h"

using namespace dolfin;

//----------------------------------------------------------------------------
Plotter::Plotter(boost::shared_ptr<const Variable> obj, QWidget *parent)
  : VTKPlotter(obj, new PlotWidget(parent))
{
  init();
}
//----------------------------------------------------------------------------
Plotter::Plotter(boost::shared_ptr<const Expression> e, boost::shared_ptr<const Mesh> m, QWidget *parent)
  : VTKPlotter(e, m, new PlotWidget(parent))
{
  init();
}
//----------------------------------------------------------------------------
bool Plotter::key_pressed(int modifiers, char key, std::string keysym)
{
  switch (modifiers + key)
  {
    case CONTROL + 'w':
      // Close window; ignore (or pass to parent widget?)
      return true;
  }

  return VTKPlotter::key_pressed(modifiers, key, keysym);
}
//----------------------------------------------------------------------------
void Plotter::init()
{
  cur_cell = -1;

  // Use a cell picker (default is prop picker)
  //vtk_pipeline->get_interactor()->SetPicker(vtkSmartPointer<vtkCellPicker>::New());

  // Receive cursor-position signals
  get_widget()->setMouseTracking(true);
  connect(get_widget(), SIGNAL(mouseMoved(int,int)), SLOT(receiveMouseMoved(int,int)));
  connect(get_widget(), SIGNAL(mouseClick(int,int)), SLOT(receiveMousePress(int,int)));

  // Prevent window move/resize
  parameters["tile_windows"] = false;
}
//----------------------------------------------------------------------------
void Plotter::receiveMouseMoved(int x, int y)
{
  const QSize size = get_widget()->size();

  vtkNew<vtkCellPicker> picker;
  if (picker->Pick(x, size.height()-y-1, 0, vtk_pipeline->get_renderer()))
  {
    cur_cell = picker->GetCellId();
    double *c = picker->GetMapperPosition();
    emit worldPos(c[0], c[1], c[2]);
  }
  else
  {
    cur_cell = -1;
  }

  emit cellId(cur_cell);
}
//----------------------------------------------------------------------------
void Plotter::receiveMousePress(int x, int y)
{
  if (cur_cell >= 0)
  {
    emit cellPicked(cur_cell);
  }
}
//----------------------------------------------------------------------------
void Plotter::toggleMesh()
{
  // FIXME: Lazy + ugly
  VTKPlotter::key_pressed(0, 'm', "m");
}
//----------------------------------------------------------------------------
void Plotter::update()
{
  VTKPlotter::plot();
}
//----------------------------------------------------------------------------
