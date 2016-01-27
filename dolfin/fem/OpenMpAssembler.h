// Copyright (C) 2007-2011 Anders Logg
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
// Modified by Garth N. Wells, 2007-2008.
// Modified by Ola Skavhaug, 2008.
//
// First added:  2007-01-17
// Last changed: 2011-09-29

#ifndef __OPENMP_ASSEMBLER_H
#define __OPENMP_ASSEMBLER_H

#ifdef HAS_OPENMP

#include <vector>
#include "AssemblerBase.h"

namespace dolfin
{

  // Forward declarations
  class GenericTensor;
  class Form;
  class UFC;
  template<typename T> class MeshFunction;

  /// This class provides automated assembly of linear systems, or
  /// more generally, assembly of a sparse tensor from a given
  /// variational form.
  ///
  /// The MeshFunction arguments can be used to specify assembly over
  /// subdomains of the mesh cells, exterior facets or interior
  /// facets. Either a null pointer or an empty MeshFunction may be
  /// used to specify that the tensor should be assembled over the
  /// entire set of cells or facets.

  class OpenMpAssembler : public AssemblerBase
  {
  public:

    /// Constructor
    OpenMpAssembler() {}

    /// Assemble tensor from given form
    void assemble(GenericTensor& A, const Form& a);

  private:

    // Assemble over cells
    void assemble_cells(GenericTensor& A, const Form& a, UFC& ufc,
                        std::shared_ptr<const MeshFunction<std::size_t>> domains,
                        std::vector<double>* values);

    // Assemble over exterior facets
    void assemble_cells_and_exterior_facets(GenericTensor& A,
             const Form& a, UFC& ufc,
             std::shared_ptr<const MeshFunction<std::size_t>> cell_domains,
             std::shared_ptr<const MeshFunction<std::size_t>> exterior_facet_domains,
             std::vector<double>* values);

    // Assemble over interior facets
    void assemble_interior_facets(GenericTensor& A, const Form& a, UFC& ufc,
             std::shared_ptr<const MeshFunction<std::size_t>> domains,
             std::shared_ptr<const MeshFunction<std::size_t>> cell_domains,
             std::vector<double>* values);

  };

}

#endif
#endif
