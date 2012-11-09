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

#ifndef __PLOT_WIDGET_H
#define __PLOT_WIDGET_H

#include <QVTKWidget.h>

class PlotWidget : public QVTKWidget
{
  Q_OBJECT

  /// Extends QVTKWidget to send signals on mouse move and click.

public:

  PlotWidget(QWidget *parent=NULL);

protected:

  virtual void mouseMoveEvent(QMouseEvent *);

  virtual void mousePressEvent(QMouseEvent *);

  virtual void mouseReleaseEvent(QMouseEvent *);

signals:

  void mouseMoved(int x, int y);

  void mouseClick(int x, int y);

private:

  // Used to decide which mouse event is a click
  bool button1_click_in_progress;

};

#endif
