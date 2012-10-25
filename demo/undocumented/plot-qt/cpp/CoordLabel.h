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
// First added:  2012-09-18
// Last changed: 2012-09-18

#ifndef __COORD_LABEL_H
#define __COORD_LABEL_H

#include <QLabel>

class CoordLabel : public QLabel
{
  Q_OBJECT

  /// A simple wrapper around QLabel, to create simple gui elements for
  /// formatted display of numbers. Add setNum() slots as required.

public:

  CoordLabel(const char *format, QWidget *parent=NULL);

public slots:

  void setNum(int);
  void setNum(int,int);
  void setNum(double,double,double);

private:

  const char *_format;

};

#endif
