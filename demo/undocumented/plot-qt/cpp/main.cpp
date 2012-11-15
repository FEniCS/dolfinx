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
//
// This demo illustrates embedding the plot window in a Qt application.

#include <QtGui>

#include <dolfin.h>

#include "CoordLabel.h"
#include "BoundaryMeshFunction.h"
#include "Plotter.h"

using namespace dolfin;

//----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  if (getenv("DOLFIN_NOPLOT"))
  {
    warning("DOLFIN_NOPLOT is set; not running %s", argv[0]);
    return 0;
  }

  // Create application and top-level window
  QApplication app(argc, argv);
  QWidget window;
  window.setWindowTitle("Qt embedded plot window demo");

  // Create plotter
  UnitCube unit_cube(4, 4, 4);
  BoundaryMeshFunction meshfunc(unit_cube);
  Plotter plotter(reference_to_no_delete_pointer(meshfunc));
  plotter.parameters["range_min"] = 0.0;
  plotter.parameters["range_max"] = 1.0;
  plotter.parameters["scalarbar"] = false;

  // All keyboard events are handled by the plotter
  plotter.get_widget()->grabKeyboard();

  // Create bottom row of labels/buttons
  CoordLabel *pixel_x_label = new CoordLabel("Pixel: (%d,%d)");
  CoordLabel *cell_label    = new CoordLabel("Cell: %d");
  CoordLabel *world_x_label = new CoordLabel("Coordinate: (%.2f,%.2f,%.2f)");
  QPushButton *toggle       = new QPushButton("Toggle mesh");

  QBoxLayout *sublayout = new QHBoxLayout();
  sublayout->addWidget(pixel_x_label);
  sublayout->addWidget(cell_label);
  sublayout->addWidget(world_x_label);
  sublayout->addWidget(toggle);

  // Create main layout (the plot window above the row of labels/buttons)
  QBoxLayout *layout = new QVBoxLayout();
  layout->addWidget(plotter.get_widget());
  layout->addLayout(sublayout);
  window.setLayout(layout);

  // Connect the plotter with the labels and buttons
  QObject::connect(plotter.get_widget(), SIGNAL(mouseMoved(int,int)), pixel_x_label, SLOT(setNum(int,int)));
  QObject::connect(&plotter, SIGNAL(cellId(int)), cell_label, SLOT(setNum(int)));
  QObject::connect(&plotter, SIGNAL(worldPos(double,double,double)), world_x_label, SLOT(setNum(double,double,double)));
  QObject::connect(toggle, SIGNAL(pressed()), &plotter, SLOT(toggleMesh()));

  // Connect the cell-pick signal to the plotted object and renderer
  QObject::connect(&plotter, SIGNAL(cellPicked(int)), &meshfunc, SLOT(toggleCell(int)));
  QObject::connect(&plotter, SIGNAL(cellPicked(int)), &plotter, SLOT(update()));

  // Set window size and show window
  // FIXME: The plot window isn't correctly sized unless plot() has been called
  // before resize().
  plotter.plot();
  window.resize(700,500);
  window.show();

  // Enter main event loop
  return app.exec();
}
//----------------------------------------------------------------------------
