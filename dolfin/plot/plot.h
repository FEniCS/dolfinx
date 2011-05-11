// Copyright (C) 2007-2009 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2007-05-02
// Last changed: 2009-10-07

#ifndef __PLOT_H
#define __PLOT_H

#include <string>
#include <dolfin/common/types.h>
#include <dolfin/mesh/MeshFunction.h>

namespace dolfin
{

  class Function;
  class Expression;
  class Mesh;

  /// Simple built-in plot commands for plotting functions and meshes.
  /// For plotting to work, PyDOLFIN and Viper must be installed.

  /// Plot function
  void plot(const Function& v,
            std::string title="Function", std::string mode="auto");

  /// Plot function
  void plot(const Expression& v, const Mesh& mesh,
            std::string title="Expression", std::string mode="auto");

  /// Plot mesh
  void plot(const Mesh& mesh,
            std::string title="Mesh");

  /// Plot mesh function
  void plot(const MeshFunction<uint>& f,
            std::string title="DOLFIN MeshFunction<uint>");

  /// Plot mesh function
  void plot(const MeshFunction<double>& f,
            std::string title="MeshFunction<double>");

  /// Plot mesh function
  void plot(const MeshFunction<bool>& f,
            std::string title="MeshFunction<bool>");

}

#endif
