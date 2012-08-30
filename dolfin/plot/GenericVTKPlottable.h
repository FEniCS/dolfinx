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
// Modified by Joachim B Haga 2012
//
// First added:  2012-06-20
// Last changed: 2012-08-30

#ifndef __GENERIC_VTK_PLOTTABLE_H
#define __GENERIC_VTK_PLOTTABLE_H

#ifdef HAS_VTK

#include <vtkSmartPointer.h>
#include <vtkAlgorithmOutput.h>
#include <vtkActor2D.h>

#include <dolfin/common/Variable.h>

namespace dolfin
{

  class Parameters;

  /// This class defines a common interface for objects that can be plotted by
  /// the VTKPlotter class

  class GenericVTKPlottable
  {
  public:

    /// Initialize the parts of the pipeline that this class controls
    virtual void init_pipeline(const Parameters& parameters) = 0;

    /// Update the plottable data. The variable may be empty, or it may be a
    /// new variable to plot. is_compatible(var) must be true.
    virtual void update(boost::shared_ptr<const Variable> var, const Parameters& parameters, int framecounter) = 0;

    /// Return whether this plottable is compatible with the variable
    virtual bool is_compatible(const Variable &var) const =0;

    /// Update the scalar range of the plottable data
    virtual void update_range(double range[2]) = 0;

    /// Return geometric dimension
    virtual uint dim() const = 0;

    /// Return true if depth sorting is required
    virtual bool requires_depthsort() const = 0;

    /// Return data to visualize
    virtual vtkSmartPointer<vtkAlgorithmOutput> get_output() const = 0;

    /// Get an actor for showing vertex labels
    virtual vtkSmartPointer<vtkActor2D> get_vertex_label_actor()
    {
      warning("Plotting of vertex labels is not implemented by the current"
          " VTK plottable type.");
      // Return empty actor to have something (invisible) to render
      return vtkSmartPointer<vtkActor2D>::New();
    }

  };
}

#endif

#endif
