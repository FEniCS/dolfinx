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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Fredrik Valdmanis, 2012
//
// First added:  2007-05-02
// Last changed: 2012-06-14

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

  // FIXME: What to do with the old default arguments? The "auto" mode is no
  // longer used by the plotter.

  /// Plot function
  void plot(const Function& function,
            std::string title="Function", std::string mode="auto");

  /// Plot function (shared_ptr version)
  void plot(boost::shared_ptr<const Function> function,
            std::string title="Function", std::string mode="auto");

  /// Plot expression
  void plot(const Expression& expression,
            const Mesh& mesh,
            std::string title="Expression", std::string mode="auto");

  /// Plot expression (shared_ptr version)
  void plot(boost::shared_ptr<const Expression> expression,
            boost::shared_ptr<const Mesh> mesh,
            std::string title="Expression", std::string mode="auto");

  /// Plot mesh
  void plot(const Mesh& mesh,
            std::string title="Mesh");

  /// Plot mesh (shared_ptr version)
  void plot(boost::shared_ptr<const Mesh> mesh,
            std::string title="Mesh");

  /// Plot Dirichlet BC
  void plot(const DirichletBC& bc,
            std::string title="Dirichlet B.C.");

  /// Plot Dirichlet BC (shared_ptr version)
  void plot(boost::shared_ptr<const DirichletBC> bc,
            std::string title="Dirichlet B.C.");

  /// Plot uint-valued mesh function
  void plot(const MeshFunction<uint>& mesh_function,
            std::string title="DOLFIN MeshFunction<uint>");

  /// Plot uint-valued mesh function (shared_ptr version)
  void plot(boost::shared_ptr<const MeshFunction<uint> > mesh_function,
            std::string title="DOLFIN MeshFunction<uint>");

  /// Plot double-valued mesh function
  void plot(const MeshFunction<double>& mesh_function,
            std::string title="MeshFunction<double>");

  /// Plot double-valued mesh function  (shared_ptr version)
  void plot(boost::shared_ptr<const MeshFunction<double> > mesh_function,
            std::string title="MeshFunction<double>");

  /// Plot boolean-valued mesh function
  void plot(const MeshFunction<bool>& mesh_function,
            std::string title="MeshFunction<bool>");

  /// Plot boolean-valued mesh function (shared_ptr version)
  void plot(boost::shared_ptr<const MeshFunction<bool> > mesh_function,
            std::string title="MeshFunction<bool>");

  /// Make the current plot interactive
  void interactive();

}

#endif
