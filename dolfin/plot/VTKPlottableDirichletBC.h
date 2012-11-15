// Copyright (C) 2012 Joachim B Haga
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
// First added:  2012-08-27
// Last changed: 2012-09-11

#ifndef __VTK_PLOTTABLE_DIRICHLETBC_H
#define __VTK_PLOTTABLE_DIRICHLETBC_H

#ifdef HAS_VTK

#include "VTKPlottableGenericFunction.h"

namespace dolfin
{

  class DirichletBC;
  class Function;

  /// Data wrapper class for Dirichlet boundary conditions

  class VTKPlottableDirichletBC : public VTKPlottableGenericFunction
  {
  public:

    explicit
    VTKPlottableDirichletBC(boost::shared_ptr<const DirichletBC> bc);

    /// Additional parameters for VTKPlottableDirichletBC
    virtual Parameters default_parameters()
    {
      return Parameters();
    }

    /// Initialize the parts of the pipeline that this class controls
    virtual void init_pipeline(const Parameters& parameters);

    /// Check if the plotter is compatible with a given DirichletBC variable
    bool is_compatible(const Variable &var) const;

    /// Update the scalar range of the plottable data
    void update(boost::shared_ptr<const Variable> var, const Parameters& parameters, int framecounter);

  private:

    boost::shared_ptr<const DirichletBC> _bc;
  };

  VTKPlottableDirichletBC *CreateVTKPlottable(boost::shared_ptr<const DirichletBC>);

}

#endif

#endif
