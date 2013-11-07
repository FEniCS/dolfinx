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
// Last changed: 2013-11-07

#ifndef __DOLFIN_ASSIGN_H
#define __DOLFIN_ASSIGN_H

#include <vector>
#include <boost/shared_ptr.hpp>

namespace dolfin
{
  
  class Function;

  void assign(boost::shared_ptr<Function> receiving_func,
	      boost::shared_ptr<const Function> assigning_func);

  void assign(boost::shared_ptr<Function> receiving_func,
	      std::vector<boost::shared_ptr<const Function> > assigning_funcs);

  void assign(std::vector<boost::shared_ptr<Function> > receiving_funcs, 
	      boost::shared_ptr<const Function> assigning_func);

}

#endif
