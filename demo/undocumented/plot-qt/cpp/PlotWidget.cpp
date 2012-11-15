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

#include "PlotWidget.h"
#include <QMouseEvent>

//----------------------------------------------------------------------------
PlotWidget::PlotWidget(QWidget *parent)
  : QVTKWidget(parent)
{
  setMouseTracking(true);
}
//----------------------------------------------------------------------------
void PlotWidget::mouseMoveEvent(QMouseEvent *event)
{
  emit mouseMoved(event->x(), event->y());
  button1_click_in_progress = false;
  QVTKWidget::mouseMoveEvent(event);
}
//----------------------------------------------------------------------------
void PlotWidget::mousePressEvent(QMouseEvent *event)
{
  button1_click_in_progress = (event->buttons() == Qt::LeftButton);
  QVTKWidget::mousePressEvent(event);
}
//----------------------------------------------------------------------------
void PlotWidget::mouseReleaseEvent(QMouseEvent *event)
{
  if (button1_click_in_progress && event->buttons() == Qt::NoButton)
  {
    emit mouseClick(event->x(), event->y());
  }
  QVTKWidget::mouseReleaseEvent(event);
}
//----------------------------------------------------------------------------

