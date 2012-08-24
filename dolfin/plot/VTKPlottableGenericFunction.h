// Copyright (C) 2012 Fredrik Valdmanis
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
// First added:  2012-06-20
// Last changed: 2012-06-26

#ifndef __VTK_PLOTTABLE_GENERIC_FUNCTION_H
#define __VTK_PLOTTABLE_GENERIC_FUNCTION_H

#ifdef HAS_VTK

#include <boost/shared_ptr.hpp>
#include <vtkWarpScalar.h>
#include <vtkWarpVector.h>
#include <vtkGlyph3D.h>

#include "VTKPlottableMesh.h"

namespace dolfin
{

  // Forward declarations
  class Expression;
  class Function;
  class GenericFunction;
  class GenericVTKPlottable;
  class Mesh;
  class Parameters;
  class VTKPlottableMesh;

  /// Data wrapper class for plotting generic functions, including
  /// instances of the Function and Expression classes.

  class VTKPlottableGenericFunction : public VTKPlottableMesh
  {
  public:

    explicit
    VTKPlottableGenericFunction(boost::shared_ptr<const Function> function);

    explicit
    VTKPlottableGenericFunction(boost::shared_ptr<const Expression> expression,
                                boost::shared_ptr<const Mesh> mesh);

    //--- Implementation of the GenericVTKPlottable interface ---

    /// Initialize the parts of the pipeline that this class controls
    void init_pipeline();

    /// Update the plottable data
    void update(const Parameters& parameters, int framecounter);

    /// Update the scalar range of the plottable data
    void update_range(double range[2]);

    /// Return data to visualize
    vtkSmartPointer<vtkAlgorithmOutput> get_output() const;

  private:

    // Update scalar values
    void update_scalar();

    // Update vector values
    void update_vector();

    // The function to visualize
    boost::shared_ptr<const GenericFunction> _function;

    // The scalar warp filter
    vtkSmartPointer<vtkWarpScalar> _warpscalar;

    // The vector warp filter
    vtkSmartPointer<vtkWarpVector> _warpvector;

    // The glyph filter
    vtkSmartPointer<vtkGlyph3D> _glyphs;

    // Mode flag
    std::string _mode;
  };

}

#endif

#endif
