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
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Expression.h>
#include <dolfin/io/File.h>
#include <dolfin/log/log.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "ExpressionWrapper.h"
#include "VTKPlotter.h"
#include "plot.h"

using namespace dolfin;

// Template function for getting already instantiated VTKPlotter for
// the given object. If none is found, a new one is created.
template <typename T>
VTKPlotter* get_plotter(boost::shared_ptr<const T> t)
{
  log(TRACE, "Looking for cached VTKPlotter.");

  for (std::list<VTKPlotter*>::iterator it = VTKPlotter::plotter_cache.begin(); it != VTKPlotter::plotter_cache.end(); it++)
  {
    if ( (*it)->id() == t->id() )
    {
      log(TRACE, "Found cached VTKPlotter.");
      return *it;
    }
  }

  // No previous plotter found, so we create a new one
  log(TRACE, "No VTKPlotter found in cache, creating new plotter.");
  VTKPlotter* plotter = new VTKPlotter(t);

  return plotter;
}
//-----------------------------------------------------------------------------
// Template function for plotting objects
template <typename T>
VTKPlotter* plot_object(boost::shared_ptr<const T> t,
    boost::shared_ptr<const Parameters> parameters)
{
  // Get plotter from cache
  VTKPlotter* plotter = get_plotter(t);
  dolfin_assert(plotter);

  // Set plotter parameters
  plotter->parameters.update(*parameters);

  // Plot
  plotter->plot();

  return plotter;
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const Function& function,
					   std::string title,
					   std::string mode)
{
  return plot(reference_to_no_delete_pointer(function), title, mode);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const Function> function,
					   std::string title, std::string mode)
{
  Parameters parameters;
  parameters.add("title", title);
  parameters.add("mode", mode);
  return plot(function, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const Function& function,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(function),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const Function> function,
					   boost::shared_ptr<const Parameters> parameters)
{
  dolfin_assert(function->function_space()->mesh());
  return plot_object(function, parameters);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const Expression& expression,
					   const Mesh& mesh,
					   std::string title, std::string mode)
{
  return plot(reference_to_no_delete_pointer(expression),
	      reference_to_no_delete_pointer(mesh), title, mode);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const Expression> expression,
					   boost::shared_ptr<const Mesh> mesh,
					   std::string title, std::string mode)
{
  Parameters parameters;
  parameters.add("title", title);
  parameters.add("mode", mode);
  return plot(expression, mesh, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const Expression& expression, const Mesh& mesh,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(expression),
	      reference_to_no_delete_pointer(mesh),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const Expression> expression,
					   boost::shared_ptr<const Mesh> mesh,
					   boost::shared_ptr<const Parameters> parameters)
{
  boost::shared_ptr<const ExpressionWrapper>
    e(new ExpressionWrapper(expression, mesh));
  return plot_object(e, parameters);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const Mesh& mesh,
					   std::string title)
{
  return plot(reference_to_no_delete_pointer(mesh), title);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const Mesh> mesh,
					   std::string title)
{
  Parameters parameters;
  parameters.add("title", title);
  return plot(mesh, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const Mesh& mesh,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(mesh),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const Mesh> mesh,
					   boost::shared_ptr<const Parameters> parameters)
{
  return plot_object(mesh, parameters);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const DirichletBC& bc,
					   std::string title)
{
  return plot(reference_to_no_delete_pointer(bc), title);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const DirichletBC> bc,
					   std::string title)
{
  Parameters parameters;
  parameters.add("title", title);
  return plot(bc, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const DirichletBC& bc, const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(bc),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const DirichletBC> bc,
					   boost::shared_ptr<const Parameters> parameters)
{
  return plot_object(bc, parameters);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const MeshFunction<uint>& mesh_function,
					   std::string title)
{
  return plot(reference_to_no_delete_pointer(mesh_function), title);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const MeshFunction<uint> > mesh_function,
					   std::string title)
{
  Parameters parameters;
  parameters.add("title", title);
  return plot(mesh_function, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const MeshFunction<uint>& mesh_function,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(mesh_function),
       reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const MeshFunction<uint> > mesh_function,
					   boost::shared_ptr<const Parameters> parameters)
{
  return plot_object(mesh_function, parameters);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const MeshFunction<int>& mesh_function,
					   std::string title)
{
  return plot(reference_to_no_delete_pointer(mesh_function), title);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const MeshFunction<int> > mesh_function,
					   std::string title)
{
  Parameters parameters;
  parameters.add("title", title);
  return plot(mesh_function, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const MeshFunction<int>& mesh_function,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(mesh_function),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const MeshFunction<int> > mesh_function,
					   boost::shared_ptr<const Parameters> parameters)
{
  return plot_object(mesh_function, parameters);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const MeshFunction<double>& mesh_function,
					   std::string title)
{
  return plot(reference_to_no_delete_pointer(mesh_function), title);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const MeshFunction<double> > mesh_function,
					   std::string title)
{
  Parameters parameters;
  parameters.add("title", title);
  return plot(mesh_function, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const MeshFunction<double>& mesh_function,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(mesh_function),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const MeshFunction<double> > mesh_function,
					   boost::shared_ptr<const Parameters> parameters)
{
  return plot_object(mesh_function, parameters);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const MeshFunction<bool>& mesh_function,
					   std::string title)
{
  return plot(reference_to_no_delete_pointer(mesh_function), title);
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const MeshFunction<bool> > mesh_function,
					   std::string title)
{
  Parameters parameters;
  parameters.add("title", title);
  return plot(mesh_function, reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(const MeshFunction<bool>& mesh_function,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(mesh_function),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
VTKPlotter* dolfin::plot(boost::shared_ptr<const MeshFunction<bool> > mesh_function,
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
    // Prepare interactiveness on every plotter
    for (std::list<VTKPlotter*>::iterator it = VTKPlotter::plotter_cache.begin(); it != VTKPlotter::plotter_cache.end(); it++)
    {
      (*it)->interactive(false);
    }

    (*VTKPlotter::plotter_cache.begin())->start_eventloop();
  }
}
//-----------------------------------------------------------------------------
