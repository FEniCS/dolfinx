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
// Last changed: 2015-11-11

#ifndef __PLOT_H
#define __PLOT_H

#include <string>
#include <memory>

namespace dolfin
{

  // Forward declarations
  class Variable;
  class Expression;
  class Mesh;
  class MultiMesh;
  class Parameters;
  class VTKPlotter;

  /// Make the current plots interactive. If really is set, the interactive
  /// mode is entered even if 'Q' has been pressed.
  void interactive(bool really=false);

  //---------------------------------------------------------------------------
  /// Simple built-in plot commands for plotting functions and meshes
  //---------------------------------------------------------------------------

  /// Plot variable of any supported type
  void plot(const Variable&, std::string title="", std::string mode="auto");

  /// Plot variable (shared_ptr version)
  std::shared_ptr<VTKPlotter> plot(std::shared_ptr<const Variable>,
                                   std::string title="",
                                   std::string mode="auto");

  /// Plot variable (parameter version)
  void plot(const Variable&, const Parameters& parameters);

  /// Plot variable (parameter, shared_ptr version)
  std::shared_ptr<VTKPlotter> plot(std::shared_ptr<const Variable>,
                                   std::shared_ptr<const Parameters> parameters);

  //---------------------------------------------------------------------------
  // Specialized versions for Expression together with Mesh
  //---------------------------------------------------------------------------

  /// Plot expression
  void plot(const Expression& expression, const Mesh& mesh,
            std::string title="", std::string mode="auto");

  /// Plot expression (shared_ptr version)
  std::shared_ptr<VTKPlotter> plot(std::shared_ptr<const Expression> expression,
                                   std::shared_ptr<const Mesh> mesh,
                                   std::string title="",
                                   std::string mode="auto");

  /// Plot expression (parameter version)
  void plot(const Expression& expression, const Mesh& mesh,
            const Parameters& parameters);

  /// Plot expression (parameter, shared_ptr version)
  std::shared_ptr<VTKPlotter> plot(std::shared_ptr<const Expression> expression,
                                   std::shared_ptr<const Mesh> mesh,
                                   std::shared_ptr<const Parameters> parameters);

  //---------------------------------------------------------------------------
  // Specialized utility functions for plotting
  //---------------------------------------------------------------------------

  // FIXME: This is very peculiar code. Why is there are shared_ptr
  // version when the function does not return an object?

  // Plot multimesh
  void plot(const MultiMesh& multimesh);

  // Plot multimesh (shared_ptr version)
  void plot(std::shared_ptr<const MultiMesh> multimesh);


}

#endif
