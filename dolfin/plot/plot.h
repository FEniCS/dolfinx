// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-02
// Last changed: 2009-03-16

#ifndef __PLOT_H
#define __PLOT_H

#include <string>
#include <dolfin/common/types.h>
#include <dolfin/mesh/MeshFunction.h>

namespace dolfin
{

  class Function;
  class Mesh;

  /// Simple built-in plot commands for plotting functions and meshes.
  /// For plotting to work, PyDOLFIN and Viper must be installed.

  /// Plot function
  void plot(const Function& v, std::string title="Untitled Function", std::string mode="auto");

  /// Plot mesh
  void plot(const Mesh& mesh, std::string title="Untitled Mesh");

  /// Plot mesh function
  void plot(const MeshFunction<uint>& f, std::string title="Untitled MeshFunction");

  /// Plot mesh function
  void plot(const MeshFunction<double>& f, std::string title="Untitled MeshFunction");

  /// Plot mesh function
  void plot(const MeshFunction<bool>& f, std::string title="Untitled MeshFunction");

}

#endif
