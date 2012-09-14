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
// Last changed: 2012-09-14

#include "Plotter.h"

using namespace dolfin;

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

void Plotter::init()
{
  // Prevent window move/resize
  parameters["tile_windows"] = false;

  get_widget()->setMouseTracking(true);
}

void Plotter::toggleMesh()
{
  // FIXME: Lazy + ugly
  VTKPlotter::key_pressed(0, 'm', "m");
}
