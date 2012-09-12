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
// Modified by Joachim Berdal Haga, 2008, 2012.
// Modified by Garth N. Wells, 2008.
// Modified by Benjamin Kehlet, 2012
//
// First added:  2007-05-02
// Last changed: 2012-09-12

#include <cstdlib>
#include <sstream>
#include <list>

#include <dolfin/common/utils.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Expression.h>
#include <dolfin/generation/CSGGeometry.h>
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
static std::list<boost::shared_ptr<VTKPlotter> > stored_plotters;

//-----------------------------------------------------------------------------
// Template function for getting already instantiated VTKPlotter for
// the given object. If none is found, a new one is created.
template <class T>
boost::shared_ptr<VTKPlotter> get_plotter(boost::shared_ptr<T> t, std::string key)
{
  log(TRACE, "Looking for cached VTKPlotter [%s].", key.c_str());

  for (std::list<boost::shared_ptr<VTKPlotter> >::iterator it = stored_plotters.begin(); it != stored_plotters.end(); it++)
  {
    if ( (*it)->key() == key && (*it)->is_compatible(t) )
    {
      log(TRACE, "Found compatible cached VTKPlotter.");
      return *it;
    }
  }

  // No previous plotter found, so create a new one
  log(TRACE, "No VTKPlotter found in cache, creating new plotter.");
  boost::shared_ptr<VTKPlotter> plotter(new VTKPlotter(t));
  plotter->set_key(key);
  stored_plotters.push_back(plotter);

  return plotter;
}
//-----------------------------------------------------------------------------
// Template function for plotting objects
template <class T>
boost::shared_ptr<VTKPlotter> plot_object(boost::shared_ptr<T> t,
					  boost::shared_ptr<const Parameters> parameters,
                                          std::string key)
{
  // Get plotter from cache. Key given as parameter takes precedence.
  const Parameter *param_key = parameters->find_parameter("key");
  if (param_key && param_key->is_set())
  {
    key = (std::string)*param_key;
  }

  boost::shared_ptr<VTKPlotter> plotter = get_plotter(t, key);

  // Set plotter parameters
  plotter->parameters.update(*parameters);

  // Plot
  plotter->plot(t);

  return plotter;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Parameters> default_parameters(std::string title,
                                                 std::string mode="auto")
{
  boost::shared_ptr<Parameters> parameters(new Parameters());
  parameters->add("title", title);
  parameters->add("mode", mode);
  return parameters;
}
//-----------------------------------------------------------------------------
void dolfin::interactive(bool really)
{
  VTKPlotter::all_interactive(really);
}
//-----------------------------------------------------------------------------
template <class T>
boost::shared_ptr<VTKPlotter> dolfin::plot(const T& t,
					   std::string title,
					   std::string mode)
{
  return plot(reference_to_no_delete_pointer(t), title, mode);
}
//-----------------------------------------------------------------------------
template <class T>
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<T> t,
					   std::string title, std::string mode)
{
  return plot(t, default_parameters(title, mode));
}
//-----------------------------------------------------------------------------
template <class T>
boost::shared_ptr<VTKPlotter> dolfin::plot(const T& t,
					   const Parameters& parameters)
{
  return plot(reference_to_no_delete_pointer(t),
	      reference_to_no_delete_pointer(parameters));
}
//-----------------------------------------------------------------------------
template <class T>
boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<T> t,
					   boost::shared_ptr<const Parameters> parameters)
{
  return plot_object(t, parameters, VTKPlotter::to_key(*t));
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
  return plot(expression, mesh, default_parameters(title, mode));
}
//-----------------------------------------------------------------------------
boost::shared_ptr<VTKPlotter> dolfin::plot(const Expression& expression,
                                           const Mesh& mesh,
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
    wrapper(new ExpressionWrapper(expression, mesh));
  return plot_object(wrapper, parameters, VTKPlotter::to_key(*expression));
}

//---------------------------------------------------------------------------
// Instantiate function templates for valid types
//---------------------------------------------------------------------------

// Must instantiate both const and non-const shared_ptr<T>s, no implicit conversion; see
// http://stackoverflow.com/questions/5600150/c-template-instantiation-with-shared-ptr-to-const-t
#define INSTANTIATE(T)                                                  \
  template boost::shared_ptr<VTKPlotter> dolfin::plot(const T&,         \
                                                      std::string, std::string); \
  template boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const T >, \
                                                      std::string, std::string); \
  template boost::shared_ptr<VTKPlotter> dolfin::plot(const T&,         \
                                                      const Parameters&); \
  template boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<const T >, \
                                                      boost::shared_ptr<const Parameters>); \
  template boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<T >, \
                                                      std::string, std::string); \
  template boost::shared_ptr<VTKPlotter> dolfin::plot(boost::shared_ptr<T >, \
                                                      boost::shared_ptr<const Parameters>);

INSTANTIATE(CSGGeometry)
INSTANTIATE(DirichletBC)
INSTANTIATE(Function)
INSTANTIATE(Mesh)
INSTANTIATE(MeshFunction<bool>)
INSTANTIATE(MeshFunction<double>)
//INSTANTIATE(MeshFunction<float>)
INSTANTIATE(MeshFunction<int>)
INSTANTIATE(MeshFunction<uint>)
