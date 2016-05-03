// Copyright (C) 2007-2014 Anders Logg and Fredrik Valdmanis
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
// Modified by Joachim Berdal Haga, 2008, 2012.
// Modified by Garth N. Wells, 2008.
// Modified by Benjamin Kehlet, 2012
//
// First added:  2007-05-02
// Last changed: 2015-11-11

#include <cstdlib>
#include <sstream>
#include <list>

#include <dolfin/common/utils.h>
#include <dolfin/mesh/MultiMesh.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Expression.h>
#include <dolfin/io/File.h>
#include <dolfin/log/log.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/parameter/Parameters.h>
#include "ExpressionWrapper.h"
#include "VTKPlotter.h"
#include "plot.h"

using namespace dolfin;

// Global list of plotters created by the plot() family of functions
// in this file.  Used to search for plotter objects in get_plotter()
// and to ensure that plotter objects are correctly destroyed when the
// program terminates.
static std::list<std::shared_ptr<VTKPlotter>> stored_plotters;

//-----------------------------------------------------------------------------
// Function for getting already instantiated VTKPlotter for
// the given object. If none is found, a new one is created.
std::shared_ptr<VTKPlotter> get_plotter(std::shared_ptr<const Variable> obj,
                                        std::string key)
{
  log(TRACE, "Looking for cached VTKPlotter [%s].", key.c_str());

  for (auto it = stored_plotters.begin(); it != stored_plotters.end(); it++)
  {
    if ( (*it)->key() == key && (*it)->is_compatible(obj) )
    {
      log(TRACE, "Found compatible cached VTKPlotter.");
      return *it;
    }
  }

  // No previous plotter found, so create a new one
  log(TRACE, "No VTKPlotter found in cache, creating new plotter.");
  std::shared_ptr<VTKPlotter> plotter(new VTKPlotter(obj));
  plotter->set_key(key);
  stored_plotters.push_back(plotter);

  return plotter;
}
//-----------------------------------------------------------------------------
// Function for plotting objects
std::shared_ptr<VTKPlotter> plot_object(std::shared_ptr<const Variable> obj,
                                        std::shared_ptr<const Parameters> p,
                                        std::string key)
{
  // Get plotter from cache. Key given as parameter takes precedence.
  const Parameter *param_key = p->find_parameter("key");
  if (param_key && param_key->is_set())
  {
    key = (std::string)*param_key;
  }

  std::shared_ptr<VTKPlotter> plotter = get_plotter(obj, key);

  // Set plotter parameters
  plotter->parameters.update(*p);

  // Plot
  plotter->plot(obj);

  return plotter;
}
//-----------------------------------------------------------------------------
std::shared_ptr<Parameters> new_parameters(std::string title, std::string mode)
{
  std::shared_ptr<Parameters> p(new Parameters());
  if (!title.empty())
  {
    p->add("title", title);
  }
  p->add("mode", mode);
  return p;
}
//-----------------------------------------------------------------------------
void dolfin::interactive(bool really)
{
  VTKPlotter::all_interactive(really);
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Variable& var, std::string title, std::string mode)
{
  plot(reference_to_no_delete_pointer(var), title, mode);
}
//-----------------------------------------------------------------------------
std::shared_ptr<VTKPlotter> dolfin::plot(std::shared_ptr<const Variable> var,
                                         std::string title, std::string mode)
{
  return plot(var, new_parameters(title, mode));
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Variable& var, const Parameters& p)
{
  plot(reference_to_no_delete_pointer(var), reference_to_no_delete_pointer(p));
}
//-----------------------------------------------------------------------------
std::shared_ptr<VTKPlotter> dolfin::plot(std::shared_ptr<const Variable> var,
                                         std::shared_ptr<const Parameters> p)
{
  return plot_object(var, p, VTKPlotter::to_key(*var));
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Expression& expression, const Mesh& mesh,
                  std::string title, std::string mode)
{
  plot(reference_to_no_delete_pointer(expression),
	      reference_to_no_delete_pointer(mesh), title, mode);
}
//-----------------------------------------------------------------------------
std::shared_ptr<VTKPlotter> dolfin::plot(std::shared_ptr<const Expression> expression,
                                         std::shared_ptr<const Mesh> mesh,
                                         std::string title, std::string mode)
{
  return plot(expression, mesh, new_parameters(title, mode));
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Expression& expression, const Mesh& mesh,
                  const Parameters& p)
{
  plot(reference_to_no_delete_pointer(expression),
       reference_to_no_delete_pointer(mesh),
       reference_to_no_delete_pointer(p));
}
//-----------------------------------------------------------------------------
std::shared_ptr<VTKPlotter> dolfin::plot(std::shared_ptr<const Expression> expression,
                                         std::shared_ptr<const Mesh> mesh,
                                         std::shared_ptr<const Parameters> p)
{
  auto wrapper = std::make_shared<const ExpressionWrapper>(expression, mesh);
  return plot_object(wrapper, p, VTKPlotter::to_key(*expression));
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MultiMesh& multimesh)
{
  plot(reference_to_no_delete_pointer(multimesh));
}
//-----------------------------------------------------------------------------
void dolfin::plot(std::shared_ptr<const MultiMesh> multimesh)
{
  dolfin_assert(multimesh);
  multimesh->_plot();
}
//-----------------------------------------------------------------------------
