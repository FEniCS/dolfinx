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
// Last changed: 2012-01-30

//=============================================================================
// Forward declarations of common types in DOLFIN needed for template
// instantiations
//=============================================================================

namespace dolfin
{

  // parameter
  class Parameters;

  // common
  template<typename T> class Hierarchical;

  // mesh
  class Mesh;
  class LocalMeshData;
  template<typename T> class MeshFunction;
  template<typename T> class MeshValueCollection;
  template<typename T> class MeshEntityIteratorBase;

  // adaptivity
  class ErrorControl;

  // fem
  class Form;
  class LinearVariationalProblem;
  class NonlinearVariationalProblem;
  class DirichletBC;
  class FiniteElement;

  // function
  class FunctionSpace;
  class Function;
  class FunctionPlotData;

  // la
  class GenericVector;
  class GenericMatrix;

}
