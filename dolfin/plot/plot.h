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
// Modified by Joachim B Haga 2012
//
// First added:  2007-05-02
// Last changed: 2012-09-12

#ifndef __PLOT_H
#define __PLOT_H

#include <string>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>

namespace dolfin
{

  // Forward declarations
  class CSGGeometry;
  class DirichletBC;
  class Function;
  class Expression;
  class Mesh;
  template<typename T> class MeshFunction;
  class Parameters;
  class VTKPlotter;

  /// Make the current plots interactive. If really is set, the interactive
  /// mode is entered even if 'Q' has been pressed.
  void interactive(bool really=false);

  //---------------------------------------------------------------------------
  /// Simple built-in plot commands for plotting functions and meshes.
  //---------------------------------------------------------------------------

  // Plot variable of any supported type
  template <class T>
  boost::shared_ptr<VTKPlotter> plot(const T& t,
				     std::string title="",
				     std::string mode="auto");

  /// Plot variable (shared_ptr version)
  template <class T>
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<T> t,
				     std::string title="",
				     std::string mode="auto");

  /// Plot variable (parameter version)
  template <class T>
  boost::shared_ptr<VTKPlotter> plot(const T& t,
				     const Parameters& parameters);

  /// Plot variable (parameter, shared_ptr version)
  template <class T>
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<T> t,
				     boost::shared_ptr<const Parameters> parameters);

  //---------------------------------------------------------------------------
  // Specialised versions for Expression together with Mesh
  //---------------------------------------------------------------------------

  /// Plot expression
  boost::shared_ptr<VTKPlotter> plot(const Expression& expression,
				     const Mesh& mesh,
				     std::string title="",
				     std::string mode="auto");

  /// Plot expression (shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const Expression> expression,
                                     boost::shared_ptr<const Mesh> mesh,
				     std::string title="",
				     std::string mode="auto");

  /// Plot expression (parameter version)
  boost::shared_ptr<VTKPlotter> plot(const Expression& expression,
				     const Mesh& mesh,
				     const Parameters& parameters);

  /// Plot expression (parameter, shared_ptr version)
  boost::shared_ptr<VTKPlotter> plot(boost::shared_ptr<const Expression> expression,
				     boost::shared_ptr<const Mesh> mesh,
				     boost::shared_ptr<const Parameters> parameters);

}

#ifdef SWIG
// Make template instantiations (in plot.cpp) available to SWIG
%template(plot) dolfin::plot<dolfin::CSGGeometry>;
%template(plot) dolfin::plot<dolfin::DirichletBC>;
%template(plot) dolfin::plot<dolfin::Function>;
%template(plot) dolfin::plot<dolfin::Mesh>;
%template(plot) dolfin::plot<dolfin::MeshFunction<bool> >;
%template(plot) dolfin::plot<dolfin::MeshFunction<double> >;
//%template(plot) dolfin::plot<dolfin::MeshFunction<float> >;
%template(plot) dolfin::plot<dolfin::MeshFunction<int> >;
%template(plot) dolfin::plot<dolfin::MeshFunction<uint> >;
#endif

#endif
