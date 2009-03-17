// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-16
// Last changed: 2009-03-16

#ifndef __FUNCTION_PLOT_DATA_H
#define __FUNCTION_PLOT_DATA_H

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/la/Vector.h>

namespace dolfin
{

  class XMLFunctionPlotData;

  class Function;

  /// This class is used for communicating plot data for functions
  /// to and from (XML) files. It is used by DOLFIN for plotting
  /// Function objects. The data is stored as a mesh and a vector
  /// of interpolated vertex values.

  class FunctionPlotData : public Variable
  {
  public:

    /// Create plot data for given function
    FunctionPlotData(const Function& v);

    /// Create empty data to be read from file
    FunctionPlotData();

    /// Destructor
    ~FunctionPlotData();

    // The mesh
    Mesh mesh;

    // The vertex values
    Vector vertex_values;

    // Value rank
    uint rank;

    typedef XMLFunctionPlotData XMLHandler;

  };

}

#endif
