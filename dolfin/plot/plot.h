// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-02
// Last changed: 2007-05-07

#ifndef __PLOT_H
#define __PLOT_H

#include <string>
#include <dolfin/main/constants.h>
#include <dolfin/mesh/MeshFunction.h>

namespace dolfin
{

  class Function;
  class Mesh;

  /// Simple built-in plot commands for plotting functions and meshes.
  /// For plotting to work, PyDOLFIN and Viper must be installed.
  /// Specifying the plotting mode is only relevant for vector-valued
  /// functions and relevant plotting modes are then "vector" (default)
  /// and "displacement".
  
  /// Plot function
  void plot(Function& f, std::string mode = "");

  /// Plot mesh
  void plot(Mesh& mesh, std::string mode = "");

  /// Plot mesh function
  void plot(MeshFunction<uint>& f, std::string mode = "");

  /// Plot mesh function
  void plot(MeshFunction<real>& f, std::string mode = "");

  /// Plot mesh function
  void plot(MeshFunction<bool>& f, std::string mode = "");

}

#endif
