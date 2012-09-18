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

#include "CoordLabel.h"

//----------------------------------------------------------------------------
CoordLabel::CoordLabel(const char *format, QWidget *parent)
  : QLabel(parent), _format(format)
{
}
//----------------------------------------------------------------------------
void CoordLabel::setNum(int x)
{
  QString txt;
  txt.sprintf(_format, x);
  setText(txt);
}
//----------------------------------------------------------------------------
void CoordLabel::setNum(int x, int y)
{
  QString txt;
  txt.sprintf(_format, x, y);
  setText(txt);
}
//----------------------------------------------------------------------------
void CoordLabel::setNum(double x, double y, double z)
{
  QString txt;
  txt.sprintf(_format, x, y, z);
  setText(txt);
}
//----------------------------------------------------------------------------
