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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Joachim Berdal Haga, 2008.
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-05-02
// Last changed: 2009-10-07

#include <stdlib.h>
#include <sstream>

#include <dolfin/common/MPI.h>
#include <dolfin/common/utils.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/io/File.h>
#include <dolfin/log/log.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/Expression.h>
#include "FunctionPlotData.h"
#include "plot.h"

using namespace dolfin;

// Template function for plotting objects
template <class T>
void plot_object(const T& t, std::string title, std::string mode)
{
  cout << "I am process " << dolfin::MPI::process_number() << endl;

  std::string prefix = parameters["plot_filename_prefix"];
  prefix += std::string("_p") + to_string(dolfin::MPI::process_number());
  const std::string filename = prefix + ".xml";
  cout << "filename = " << filename << endl;
  File file(filename);
  cout << t << endl;
  file << t;


  //if (system(("touch " + filename).c_str()) != 0)
  //  cout << "That didn't work" << endl;
}

template <class T>
void plot_object_old(const T& t, std::string title, std::string mode)
{



  if (dolfin::MPI::num_processes() > 1)
  {
    if (dolfin::MPI::process_number() == 0)
      warning("On screen plotting from C++ not yet working in parallel.");
    //return;
  }

  info("Plotting %s (%s), press 'q' to continue...",
          t.name().c_str(), t.label().c_str());

  // Get filename prefix
  std::string prefix = parameters["plot_filename_prefix"];

  for (dolfin::uint p = 0; p < dolfin::MPI::num_processes(); ++p)
  {

    if (dolfin::MPI::process_number() == p)
    {
      prefix += std::string("_p") + to_string(p);
      title += " (process " + to_string(p) + ")";

      // Save to file
      std::string filename = prefix + std::string(".xml");
      File file(filename);
      file << t;

      // Plot data from file
      std::stringstream command;
      command << "viper --mode=" << mode << " "
              << "--title=\"" << title << "\" " << filename;
      if (system(command.str().c_str()) != 0)
        warning("Unable to plot.");
    }

    dolfin::MPI::barrier();

  }
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Function& v,
                  std::string title, std::string mode)
{
  FunctionPlotData w(v, v.function_space().mesh());
  plot_object(w, title, mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Expression& v, const Mesh& mesh,
                  std::string title, std::string mode)
{
  FunctionPlotData w(v, mesh);
  plot_object(w, title, mode);
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
