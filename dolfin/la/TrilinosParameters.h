// Copyright (C) 2015 Chris Richardson
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

#ifndef __DOLFIN_TRILINOS_PARAMETERS_H
#define __DOLFIN_TRILINOS_PARAMETERS_H

#ifdef HAS_TRILINOS

#include <dolfin/parameter/Parameters.h>

#include <Teuchos_ParameterList.hpp>

namespace dolfin
{

  /// Method for translating DOLFIN Parameters to Teuchos::ParameterList
  /// needed by Trilinos objects
  class TrilinosParameters
  {

  public:

    /// Copy over parameters from a dolfin Parameters object to a Teuchos::ParameterList
    /// @param[in] params (Parameters)
    ///   dolfin parameter set
    /// @param[in,out] parameter_list (Teuchos::RCP<Teuchos::ParameterList>)
    ///   Trilinos Teuchos parameter set
    static void insert_parameters(const Parameters& params,
                             Teuchos::RCP<Teuchos::ParameterList> parameter_list);

  };

}

#endif

#endif
