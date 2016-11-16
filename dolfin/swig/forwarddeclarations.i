/* -*- C -*- */
// Copyright (C) 2012 Johan Hake
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
// First added:  2012-01-20
// Last changed: 2013-07-08

//=============================================================================
// Forward declarations of common types in DOLFIN needed for template
// instantiations
//=============================================================================

namespace dolfin
{
  // ale
  class MeshDisplacement;

  // common
  class Table;

  // parameter
  class Parameters;

  // common
  template<typename T> class Hierarchical;

  // mesh
  class IntersectionOperator;
  class Vertex;
  class Mesh;
  class Point;
  class MeshEntity;
  class LocalMeshData;
  template<typename T> class MeshFunction;
  template<typename T> class MeshValueCollection;

  // adaptivity
  class ErrorControl;

  // fem
  class Form;
  class GenericDofMap;
  class LinearVariationalProblem;
  class NonlinearVariationalProblem;
  class DirichletBC;
  class FiniteElement;

  // function
  class FunctionSpace;
  class Function;

  // la
  class GenericVector;
  class GenericMatrix;

}

//=============================================================================
// Forward declarations of common types in UFC needed for template
// instantiations
//=============================================================================

namespace ufc
{
  class cell;
  class function;
  class finite_element;
  class dofmap;
  class form;
}
