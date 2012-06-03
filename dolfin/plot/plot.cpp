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
// Modified by Joachim Berdal Haga, 2008.
// Modified by Garth N. Wells, 2008.
// Modified by Fredrik Valdmanis, 2012.
// Modified by Benjamin Kehlet, 2012
//
// First added:  2007-05-02
// Last changed: 2012-06-03

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
#include "VTKPlotter.h"
#include "plot.h"

using namespace dolfin;

std::vector<VTKPlotter> plotter_cache;

int last_used_idx = -1;

// Template function for plotting objects
template <typename T>
void plot_object(const T& t, std::string title, std::string mode)
{
  //info("Plotting %s (%s).",
  //        t.name().c_str(), t.label().c_str());
  
  // Modify title when running in parallel
  if (dolfin::MPI::num_processes() > 1)
  {
    const dolfin::uint p = dolfin::MPI::process_number();
    title += " (process " + to_string(p) + ")";
  }

#ifdef HAS_VTK
  // Maybe implement copy constructor and assignment operator in VTKPlotter?
  //VTKPlotter plotter = get_plotter(t);
  // // Set parameters!!
  //plotter.plot()
  //
  VTKPlotter plotter(t);
  plotter.plot();
#else
  dolfin_error("plot.cpp",
               "plot object",
	       "Plotting disbled. Dolfin has been compiled without VTK support");
#endif
}
//-----------------------------------------------------------------------------
// Template function for getting already instantiated VTKPlotter for the given
// object. If none is found, a new one is created and added to the cache
/*template <typename T>
VTKPlotter get_plotter(const T& t)
{
  std::vector<VTKPlotter>::const_iterator it;

  uint idx = 0;

  for(it = plotter_cache.begin(); it != plotter_cache.end(); ++it) {
    ++idx;
    if ((*it).id() == t.id()) {
      last_used_idx = idx;
      return *it;
    }
  }

  // No previous plotter found, so we create a new one
  VTKPlotter plotter(t);
  plotter_cache.push_back(plotter);
  last_used_idx = plotter_cache.size() - 1;

  return plotter;
}*/
//-----------------------------------------------------------------------------
void dolfin::plot(const Function& v,
                  std::string title, std::string mode)
{
  dolfin_assert(v.function_space()->mesh());
  plot_object(v, title, mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Expression& v, const Mesh& mesh,
                  std::string title, std::string mode)
{
  // FIXME: We should use plot_object here, but it is not possible since the
  // mesh must be explicitly passed to VTKPlotter. What to do?
#ifdef HAS_VTK
  // FIXME: Make plottable expression and do what's necessary
  PlotableExpression e(v, mesh);
  plot_object(e, title, mode);
  //VTKPlotter plotter(v, mesh);
  //plotter.parameters["title"] = title;
  //plotter.parameters["mode"] = mode;
  //plotter.plot();
#else
  dolfin_error("plot.cpp",
               "plot object",
	       "Plotting disbled. Dolfin has been compiled without VTK support");
#endif
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Mesh& mesh,
                  std::string title)
{
  plot_object(mesh, title, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<uint>& f,
                  std::string title)
{
  plot_object(f, title, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<double>& f,
                  std::string title)
{
  plot_object(f, title, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<bool>& f,
                  std::string title)
{
  plot_object(f, title, "auto");
}
//-----------------------------------------------------------------------------
/*void interactive()
{
  if (last_used_idx == -1) {
    warning("No plots have been made so far. Ignoring call to interactive().");
  } else {
    // Call interactive on the last used plotter
    plotter_cache[last_used_idx].interactive();
  }
}*/
//-----------------------------------------------------------------------------
