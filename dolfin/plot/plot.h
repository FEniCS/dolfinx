// Copyright (C) 2007-2012 Anders Logg and Fredrik Valdmanis
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
// First added:  2007-05-02
// Last changed: 2012-08-11

#ifndef __PLOT_H
#define __PLOT_H

#include <string>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>

namespace dolfin
{

  class DirichletBC;
  class Function;
  class Expression;
  class Mesh;
  template<typename T> class MeshFunction;
  class Parameters;
  class VTKPlotter;

  /// Simple built-in plot commands for plotting functions and meshes.

  /// Plot function
  boost::shared_ptr<VTKPlotter> plot(const Function& function,
				     std::string title="Function",
				     std::string mode="auto");

  /// Plot function (shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const Function> function,
				     std::string title="Function",
				     std::string mode="auto");

  /// Plot function (parameter version)
  boost::shared_ptr<VTKPlotter> plot(const Function& function,
				     const Parameters& parameters);

  /// Plot function (parameter, shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const Function> function,
				     boost::shared_ptr<const Parameters> parameters);

  /// Plot expression
  boost::shared_ptr<VTKPlotter> plot(const Expression& expression,
				     const Mesh& mesh,
				     std::string title="Expression",
				     std::string mode="auto");

  /// Plot expression (shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const Expression> expression,
		   boost::shared_ptr<const Mesh> mesh,
				     std::string title="Expression",
				     std::string mode="auto");

  /// Plot expression (parameter version)
  boost::shared_ptr<VTKPlotter> plot(const Expression& expression,
				     const Mesh& mesh,
				     const Parameters& parameters);

  /// Plot expression (parameter, shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const Expression> expression,
				     boost::shared_ptr<const Mesh> mesh,
				     boost::shared_ptr<const Parameters> parameters);

  /// Plot mesh
  boost::shared_ptr<VTKPlotter> plot(const Mesh& mesh,
				     std::string title="Mesh");

  /// Plot mesh (shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const Mesh> mesh,
				     std::string title="Mesh");

  /// Plot mesh (parameter version)
  boost::shared_ptr<VTKPlotter> plot(const Mesh& mesh,
				     const Parameters& parameters);

  /// Plot mesh (parameter, shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const Mesh> mesh,
				     boost::shared_ptr<const Parameters> parameters);

  /// Plot Dirichlet BC
  boost::shared_ptr<VTKPlotter> plot(const DirichletBC& bc,
				     std::string title="Dirichlet B.C.");

  /// Plot Dirichlet BC (shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const DirichletBC> bc,
				     std::string title="Dirichlet B.C.");

  /// Plot Dirichlet BC (parameter version)
  boost::shared_ptr<VTKPlotter> plot(const DirichletBC& bc,
				     const Parameters& parameters);

  /// Plot Dirichlet BC (parameter, shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const DirichletBC> bc,
				     boost::shared_ptr<const Parameters> parameters);

  /// Plot std::size_t-valued mesh function
  boost::shared_ptr<VTKPlotter> plot(const MeshFunction<std::size_t>& mesh_function,
				     std::string title="DOLFIN MeshFunction<unsigned int>");

  /// Plot std::size_t-valued mesh function (shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const MeshFunction<std::size_t> > mesh_function,
				     std::string title="DOLFIN MeshFunction<unsigned int>");

  /// Plot std::size_t-valued mesh function (parameter version)
  boost::shared_ptr<VTKPlotter> plot(const MeshFunction<std::size_t>& mesh_function,
				     const Parameters& parameters);

  /// Plot std::size_t-valued mesh function (parameter, shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const MeshFunction<std::size_t> > mesh_function,
				     boost::shared_ptr<const Parameters> parameters);

  /// Plot uint-valued mesh function
  boost::shared_ptr<VTKPlotter> plot(const MeshFunction<unsigned int>& mesh_function,
				     std::string title="DOLFIN MeshFunction<unsigned int>");

  /// Plot uint-valued mesh function (shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const MeshFunction<unsigned int> > mesh_function,
				     std::string title="DOLFIN MeshFunction<unsigned int>");

  /// Plot uint-valued mesh function (parameter version)
  boost::shared_ptr<VTKPlotter> plot(const MeshFunction<unsigned int>& mesh_function,
				     const Parameters& parameters);

  /// Plot uint-valued mesh function (parameter, shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const MeshFunction<unsigned int> > mesh_function,
				     boost::shared_ptr<const Parameters> parameters);

  /// Plot int-valued mesh function
  boost::shared_ptr<VTKPlotter> plot(const MeshFunction<int>& mesh_function,
				     std::string title="DOLFIN MeshFunction<int>");

  /// Plot int-valued mesh function (shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const MeshFunction<int> > mesh_function,
				     std::string title="DOLFIN MeshFunction<int>");

  /// Plot int-valued mesh function (parameter version)
  boost::shared_ptr<VTKPlotter> plot(const MeshFunction<int>& mesh_function,
				     const Parameters& parameters);

  /// Plot int-valued mesh function (parameter, shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const MeshFunction<int> > mesh_function,
				     boost::shared_ptr<const Parameters> parameters);

  /// Plot double-valued mesh function
  boost::shared_ptr<VTKPlotter> plot(const MeshFunction<double>& mesh_function,
				     std::string title="MeshFunction<double>");

  /// Plot double-valued mesh function  (shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const MeshFunction<double> > mesh_function,
				     std::string title="MeshFunction<double>");

  /// Plot double-valued mesh function  (parameter version)
  boost::shared_ptr<VTKPlotter> plot(const MeshFunction<double>& mesh_function,
				     const Parameters& parameters);

  /// Plot double-valued mesh function  (parameter, shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const MeshFunction<double> > mesh_function,
				     boost::shared_ptr<const Parameters> parameters);

  /// Plot boolean-valued mesh function
  boost::shared_ptr<VTKPlotter> plot(const MeshFunction<bool>& mesh_function,
				     std::string title="MeshFunction<bool>");

  /// Plot boolean-valued mesh function (shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const MeshFunction<bool> > mesh_function,
				     std::string title="MeshFunction<bool>");

  /// Plot boolean-valued mesh function (parameter version)
  boost::shared_ptr<VTKPlotter> plot(const MeshFunction<bool>& mesh_function,
				     const Parameters& parameters);

  /// Plot boolean-valued mesh function (parameter, shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const MeshFunction<bool> > mesh_function,
				     boost::shared_ptr<const Parameters> parameters);

  /// Make the current plot interactive
  void interactive();

}

#endif
