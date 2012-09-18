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

#ifndef __PLOTTER__H
#define __PLOTTER_H

#include <QObject>
#include <QVTKWidget.h>
#include <dolfin/plot/VTKPlotter.h>
#include "PlotWidget.h"

class Plotter : public QObject, public dolfin::VTKPlotter
{
  Q_OBJECT

  /// Extends VTKPlotter with signals and slots. Additionally adds a cell
  /// picker interface.

public:

  Plotter(boost::shared_ptr<const Variable> obj,
          QWidget *parent=NULL);

  Plotter(boost::shared_ptr<const dolfin::Expression> e,
          boost::shared_ptr<const dolfin::Mesh> m,
          QWidget *parent=NULL);

  virtual bool key_pressed(int modifiers, char key, std::string keysym);

private slots:

  void receiveMouseMoved(int x, int y);
  void receiveMousePress(int x, int y);

public slots:

  void toggleMesh();

  void update();

signals:

  void cellId(int);

  void cellPicked(int);

  void worldPos(double,double,double);

private:

  void init();

  int cur_cell;
};

#endif
