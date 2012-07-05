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
// Modified by Benjamin Kehlet, 2012
//
// First added:  2007-05-02
// Last changed: 2012-07-05

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
#include "ExpressionWrapper.h"
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

  for (dolfin::uint i = 0; i < VTKPlotter::plotter_cache.size(); i++)
  {
    dolfin_assert(VTKPlotter::plotter_cache[i]);
    if (VTKPlotter::plotter_cache[i]->id() == t->id())
    {
      log(TRACE, "Found cached VTKPlotter with index %d.", i);
      return VTKPlotter::plotter_cache[i];
    }
  }

  // No previous plotter found, so we create a new one
  log(TRACE, "No VTKPlotter found in cache, creating new plotter.");
  boost::shared_ptr<VTKPlotter> plotter(new VTKPlotter(t));

  // Adjust window position to not completely overlap previous plots
  dolfin::uint num_old_plots = VTKPlotter::plotter_cache.size();
  int width, height;
  plotter->get_window_size(width, height);

  // FIXME: This approach might need some tweaking. It tiles the winows in a
  // 2 x 2 pattern on the screen. Might not look good with more than 4 plot
  // windows
  plotter->set_window_position((num_old_plots%2)*width,
      (num_old_plots/2)*height);

  // Add plotter to cache
  VTKPlotter::plotter_cache.push_back(plotter);
  log(TRACE, "Size of plotter cache is %d.", VTKPlotter::plotter_cache.size());

  return VTKPlotter::plotter_cache.back();
}

// Template function for plotting objects
template <typename T>
boost::shared_ptr<VTKPlotter> plot_object(boost::shared_ptr<const T> t,
    boost::shared_ptr<const Parameters> parameters)
{
  // Get plotter from cache
  boost::shared_ptr<VTKPlotter> plotter = get_plotter(t);
  dolfin_assert(plotter);

  // Set plotter parameters
  plotter->parameters.update(*parameters);

  // Plot
  plotter->plot();

  return plotter;
}

//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const Function& function,
					   std::string title, 
					   std::string mode)
{
  return plot(reference_to_no_delete_pointer(function), title, mode);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const Function> function,
					   std::string title, std::string mode)
{
  Parameters parameters;
  parameters.add("title", title);
  parameters.add("mode", mode);
  return plot(function, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const Function& function, 
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(function),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const Function> function,
					   boost::shared_ptr<const Parameters> parameters)
{
  dolfin_assert(function->function_space()->mesh());
  return plot_object(function, parameters);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const Expression& expression,
					   const Mesh& mesh,
					   std::string title, std::string mode)
{
  return plot(reference_to_no_delete_pointer(expression),
	      reference_to_no_delete_pointer(mesh), title, mode);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const Expression> expression,
					   boost::shared_ptr<const Mesh> mesh,
					   std::string title, std::string mode)
{
  Parameters parameters;
  parameters.add("title", title);
  parameters.add("mode", mode);
  return plot(expression, mesh, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const Expression& expression, const Mesh& mesh,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(expression),
	      reference_to_no_delete_pointer(mesh),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const Expression> expression,
					   boost::shared_ptr<const Mesh> mesh,
					   boost::shared_ptr<const Parameters> parameters)
{
  boost::shared_ptr<const ExpressionWrapper>
    e(new ExpressionWrapper(expression, mesh));
  return plot_object(e, parameters);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const Mesh& mesh,
					   std::string title)
{
  return plot(reference_to_no_delete_pointer(mesh), title);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const Mesh> mesh,
					   std::string title)
{
  Parameters parameters;
  parameters.add("title", title);
  return plot(mesh, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const Mesh& mesh, 
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(mesh),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const Mesh> mesh,
					   boost::shared_ptr<const Parameters> parameters)
{
  return plot_object(mesh, parameters);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const DirichletBC& bc,
					   std::string title)
{
  return plot(reference_to_no_delete_pointer(bc), title);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const DirichletBC> bc,
					   std::string title)
{
  Parameters parameters;
  parameters.add("title", title);
  return plot(bc, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const DirichletBC& bc, const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(bc),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const DirichletBC> bc,
					   boost::shared_ptr<const Parameters> parameters)
{
  return plot_object(bc, parameters);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const MeshFunction<uint>& mesh_function,
					   std::string title)
{
  return plot(reference_to_no_delete_pointer(mesh_function), title);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const MeshFunction<uint> > mesh_function,
					   std::string title)
{
  Parameters parameters;
  parameters.add("title", title);
  return plot(mesh_function, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const MeshFunction<uint>& mesh_function,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(mesh_function),
       reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const MeshFunction<uint> > mesh_function,
					   boost::shared_ptr<const Parameters> parameters)
{
  return plot_object(mesh_function, parameters);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const MeshFunction<int>& mesh_function,
					   std::string title)
{
  return plot(reference_to_no_delete_pointer(mesh_function), title);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const MeshFunction<int> > mesh_function,
					   std::string title)
{
  Parameters parameters;
  parameters.add("title", title);
  return plot(mesh_function, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const MeshFunction<int>& mesh_function,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(mesh_function),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const MeshFunction<int> > mesh_function,
					   boost::shared_ptr<const Parameters> parameters)
{
  return plot_object(mesh_function, parameters);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const MeshFunction<double>& mesh_function,
					   std::string title)
{
  return plot(reference_to_no_delete_pointer(mesh_function), title);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const MeshFunction<double> > mesh_function,
					   std::string title)
{
  Parameters parameters;
  parameters.add("title", title);
  return plot(mesh_function, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const MeshFunction<double>& mesh_function,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(mesh_function),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const MeshFunction<double> > mesh_function,
					   boost::shared_ptr<const Parameters> parameters)
{
  return plot_object(mesh_function, parameters);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const MeshFunction<bool>& mesh_function,
					   std::string title)
{
  return plot(reference_to_no_delete_pointer(mesh_function), title);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const MeshFunction<bool> > mesh_function,
					   std::string title)
{
  Parameters parameters;
  parameters.add("title", title);
  return plot(mesh_function, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const MeshFunction<bool>& mesh_function,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(mesh_function),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const MeshFunction<bool> > mesh_function,
					   boost::shared_ptr<const Parameters> parameters)
{
  return plot_object(mesh_function, parameters);
}
//-----------------------------------------------------------------------------
void dolfin::interactive()
{
  if (VTKPlotter::plotter_cache.size() == 0)
  {
    warning("No plots have been shown yet. Ignoring call to interactive().");
  }
  else
  {
    // Call interactive on every plotter
    for (uint i = 0; i < VTKPlotter::plotter_cache.size(); ++i) 
    {
      VTKPlotter::plotter_cache[i]->interactive(false);
    }

    VTKPlotter::plotter_cache[VTKPlotter::plotter_cache.size()-1]->start_eventloop();
  }
}
//-----------------------------------------------------------------------------
