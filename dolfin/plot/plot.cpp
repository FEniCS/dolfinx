// Copyright (C) 2007-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Joachim Berdal Haga, 2008.
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-05-02
// Last changed: 2009-03-27

#include <stdlib.h>
#include <sstream>

#include <dolfin/parameter/parameters.h>
#include <dolfin/io/File.h>
#include "FunctionPlotData.h"
#include "plot.h"

using namespace dolfin;

// Template function for plotting objects
template <class T>
void plot_object(const T& t, std::string mode)
{
  info("Plotting %s (%s), press 'q' to continue...",
          t.name().c_str(), t.label().c_str());

  // Save to file
  const std::string filename = dolfin_get("plot file name");
  File file(filename);
  file << t;

  // Plot data from file
  std::stringstream command;
  command << "viper --mode=" << mode << " " << filename;
  if (system(command.str().c_str()) != 0)
    warning("Unable to plot.");
}

//-----------------------------------------------------------------------------
void dolfin::plot(const Function& v, std::string mode)
{
  FunctionPlotData w(v);
  plot_object(w, mode);
}
//-----------------------------------------------------------------------------
void dolfin::plot(const Mesh& mesh)
{
  plot_object(mesh, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<uint>& f)
{
  plot_object(f, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<double>& f)
{
  plot_object(f, "auto");
}
//-----------------------------------------------------------------------------
void dolfin::plot(const MeshFunction<bool>& f)
{
  plot_object(f, "auto");
}
//-----------------------------------------------------------------------------
