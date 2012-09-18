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

#ifndef __BOUNDARY_MESH_FUNCTION_H
#define __BOUNDARY_MESH_FUNCTION_H

#include <dolfin.h>
#include <QObject>

class BoundaryMeshFunction : public QObject, public dolfin::MeshFunction<bool>
{
  Q_OBJECT

  /// A MeshFunction<bool> on the boundary of a Mesh. The purpose of this class
  /// is to acceps a toggle signal, which changes the value of a single cell.

public:

  BoundaryMeshFunction(const dolfin::Mesh&);

public slots:

  void toggleCell(int);

private:

  dolfin::BoundaryMesh _bmesh;

};

#endif
