// Copyright (C) 2012 Garth N. Wells
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
// First added:  2012-03-27
// Last changed:

#ifndef __DOLFIN_STLVECTOR_H
#define __DOLFIN_STLVECTOR_H

#include "EpetraVector.h"
#include "PETScVector.h"
#include "uBLASVector.h"

namespace dolfin
{

  #ifdef HAS_PETSC
    typedef PETScVector STLVector;
  #else
   #ifdef HAS_TRILINOS
    typedef EpetraVector STLVector;
   #else
    typedef uBLASVector STLVector;
   #endif 
  #endif	

}

#endif
