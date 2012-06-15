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
// Modified by Joachim Berdal Haga, 2008.
// Modified by Garth N. Wells, 2008.
// Modified by Fredrik Valdmanis, 2012.
// Modified by Benjamin Kehlet, 2012
//
// First added:  2007-05-02
// Last changed: 2012-06-04

#include <cstdlib>
#include <sstream>

#include <dolfin/common/MPI.h>
#include <dolfin/common/utils.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/io/File.h>
#include <dolfin/log/log.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/Expression.h>
#include "PlottableExpression.h"
#include "VTKPlotter.h"
#include "plot.h"

using namespace dolfin;

// Template function for getting already instantiated VTKPlotter for
// the given object. If none is found, a new one is created and added
// to the cache.
template <typename T>
boost::shared_ptr<VTKPlotter> get_plotter(boost::shared_ptr<const T> t)
{
  log(TRACE, "Looking for cached VTKPlotter.");

  for (uint i = 0; i < VTKPlotter::plotter_cache.size(); i++)
  {
    dolfin_assert(VTKPlotter::plotter_cache[i]);
    if (VTKPlotter::plotter_cache[i]->id() == t->id())
    {
      log(TRACE, "Found cached VTKPlotter with index %d.", i);
      VTKPlotter::last_used_idx = i;
      return VTKPlotter::plotter_cache[i];
    }
  }

  // No previous plotter found, so we create a new one
  log(TRACE, "No VTKPlotter found in cache, creating new plotter.");
  boost::shared_ptr<VTKPlotter> plotter(new VTKPlotter(t));
  VTKPlotter::plotter_cache.push_back(plotter);
  VTKPlotter::last_used_idx = VTKPlotter::plotter_cache.size() - 1;
  log(TRACE, "Size of plotter cache is %d.", VTKPlotter::plotter_cache.size());

  return VTKPlotter::plotter_cache.back();
}

// Template function for plotting objects
template <typename T>
void plot_object(boost::shared_ptr<const T> t, std::string title, std::string mode)
{
#ifndef HAS_VTK
  dolfin_error("plot.cpp",
               "plot object",
	       "Plotting disbled. Dolfin has been compiled without VTK support");
#else

  // Modify title when running in parallel
  if (dolfin::MPI::num_processes() > 1)
  {
    const dolfin::uint p = dolfin::MPI::process_number();
    title += " (process " + to_string(p) + ")";
  }

  // Get plotter from cache
  boost::shared_ptr<VTKPlotter> plotter = get_plotter(t);
  dolfin_assert(plotter);

  // Set plotter parameters
  plotter->parameters["title"] = title;
  plotter->parameters["mode"] = mode;

  // Plot
  plotter->plot();

#endif
}

//-----------------------------------------------------------------------------
void dolfin::plot(const Function& function,
                  std::string title, std::string mode)
{
  plot(reference_to_no_delete_pointer(function), title, mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(boost::shared_ptr<const Function> function,
                  std::string title, std::string mode)
{
  dolfin_assert(function->function_space()->mesh());
  plot_object(function, title, mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Expression& expression,
                  const Mesh& mesh,
                  std::string title, std::string mode)
{
  plot(reference_to_no_delete_pointer(expression),
       reference_to_no_delete_pointer(mesh), title, mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(boost::shared_ptr<const Expression> expression,
                  boost::shared_ptr<const Mesh> mesh,
                  std::string title, std::string mode)
{
  boost::shared_ptr<const PlottableExpression>
    e(new PlottableExpression(expression, mesh));
  plot_object(e, title, mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Mesh& mesh,
                  std::string title)
{
  plot(reference_to_no_delete_pointer(mesh), title);
}
//-----------------------------------------------------------------------------
void dolfin::plot(boost::shared_ptr<const Mesh> mesh,
                  std::string title)
{
  plot_object(mesh, title, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const DirichletBC& bc,
                  std::string title)
{
  plot(reference_to_no_delete_pointer(bc), title);
}
//-----------------------------------------------------------------------------
void dolfin::plot(boost::shared_ptr<const DirichletBC> bc,
                  std::string title)
{
  plot_object(bc, title, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<uint>& mesh_function,
                  std::string title)
{
  plot(reference_to_no_delete_pointer(mesh_function), title);
}
//-----------------------------------------------------------------------------
void dolfin::plot(boost::shared_ptr<const MeshFunction<uint> > mesh_function,
                  std::string title)
{
  plot_object(mesh_function, title, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<double>& mesh_function,
                  std::string title)
{
  plot(reference_to_no_delete_pointer(mesh_function), title);
}
//-----------------------------------------------------------------------------
void dolfin::plot(boost::shared_ptr<const MeshFunction<double> > mesh_function,
                  std::string title)
{
  plot_object(mesh_function, title, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<bool>& mesh_function,
                  std::string title)
{
  plot(reference_to_no_delete_pointer(mesh_function), title);
}
//-----------------------------------------------------------------------------
void dolfin::plot(boost::shared_ptr<const MeshFunction<bool> > mesh_function,
                  std::string title)
{
  plot_object(mesh_function, title, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::interactive()
{
  // FIXME: The cache logic doesn't work yet
  return;
  if (VTKPlotter::plotter_cache.size() == 0) {
    warning("No plots have been made so far. Ignoring call to interactive().");
  } else {
    // Call interactive on the last used plotter
    VTKPlotter::plotter_cache[VTKPlotter::last_used_idx]->interactive();
  }
}
//-----------------------------------------------------------------------------
