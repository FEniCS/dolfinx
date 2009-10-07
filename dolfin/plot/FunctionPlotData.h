// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009.
//
// First added:  2009-03-16
// Last changed: 2009-10-07

#ifndef __FUNCTION_PLOT_DATA_H
#define __FUNCTION_PLOT_DATA_H

#include <boost/scoped_ptr.hpp>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/la/GenericVector.h>

namespace dolfin
{

  class XMLFunctionPlotData;
  class GenericFunction;

  /// This class is used for communicating plot data for functions
  /// to and from (XML) files. It is used by DOLFIN for plotting
  /// Function objects. The data is stored as a mesh and a vector
  /// of interpolated vertex values.

  class FunctionPlotData : public Variable
  {
  public:

    /// Create plot data for given function
    FunctionPlotData(const GenericFunction& v, const Mesh& mesh);

    /// Create empty data to be read from file
    FunctionPlotData();

    /// Destructor
    ~FunctionPlotData();

    /// Return vertex values
    GenericVector& vertex_values() const;

    /// The mesh
    Mesh mesh;

    /// The value rank
    uint rank;

    typedef XMLFunctionPlotData XMLHandler;

  private:

    // The vertex values
    boost::scoped_ptr<GenericVector> _vertex_values;

  };

}

#endif
