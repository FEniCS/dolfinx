// Copyright (C) 2013 Johan Hake
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
// First added:  2013-11-07
// Last changed: 2013-11-11

#ifndef __DOLFIN_ASSIGN_H
#define __DOLFIN_ASSIGN_H

#include <vector>
#include <memory>

namespace dolfin
{

  class Function;

  /// Assign one function to another. The functions must reside in the
  /// same type of FunctionSpace. One or both functions can be sub
  /// functions.
  ///
  /// @param    receiving_func (std::shared_ptr<_Function_>)
  ///         The receiving function
  /// @param    assigning_func (std::shared_ptr<_Function_>)
  ///         The assigning function
  void assign(std::shared_ptr<Function> receiving_func,
	      std::shared_ptr<const Function> assigning_func);

  /// Assign several functions to sub functions of a mixed receiving
  /// function. The number of receiving functions must sum up to the
  /// number of sub functions in the assigning mixed function. The sub
  /// spaces of the assigning mixed space must be of the same type ans
  /// size as the receiving spaces.
  /// @param    receiving_func (std::shared_ptr<_Function_>)
  ///         The receiving function
  /// @param    assigning_funcs (std::vector<std::shared_ptr<_Function_>>)
  ///         The assigning functions
  void assign(std::shared_ptr<Function> receiving_func,
	      std::vector<std::shared_ptr<const Function> > assigning_funcs);

  /// Assign sub functions of a single mixed function to single
  /// receiving functions. The number of sub functions in the
  /// assigning mixed function must sum up to the number of receiving
  /// functions. The sub spaces of the receiving mixed space must be
  /// of the same type ans size as the assigning spaces.
  /// @param    receiving_funcs (std::vector<std::shared_ptr<_Function_>>)
  ///         The receiving functions
  /// @param    assigning_func (std::shared_ptr<_Function_>)
  ///         The assigning function
  void assign(std::vector<std::shared_ptr<Function> > receiving_funcs,
	      std::shared_ptr<const Function> assigning_func);

}

#endif
