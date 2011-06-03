// Copyright (C) 2007-2009 Anders Logg
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
// Last changed: 2011-03-11

#ifndef __ASSEMBLER_H
#define __ASSEMBLER_H

#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  // Forward declarations
  class GenericTensor;
  class Form;
  class SubDomain;
  class UFC;
  template<class T> class MeshFunction;

  /// This class provides automated assembly of linear systems, or
  /// more generally, assembly of a sparse tensor from a given
  /// variational form.
  ///
  /// Subdomains for cells and facets may be specified in a number
  /// of different ways:
  ///
  /// 1. By explicitly passing MeshFunctions (as pointers) to the
  ///    assemble functions
  ///
  /// 2. By assigning subdomain indicators specified by MeshFunctions
  ///    to the Form being assembled:
  ///
  ///    form.cell_domains = cell_domains
  ///    form.exterior_facet_domains = exterior_facet_domains
  ///    form.interior_facet_domains = interior_facet_domains
  ///
  /// 3. By MeshFunctions stored in MeshData as
  ///
  ///    "cell_domains"
  ///    "exterior_facet_domains"
  ///    "interior_facet_domains"
  ///
  /// 4. By specifying a SubDomain which specifies the domain numbered
  ///    as 0 (with the rest treated as domain number 1)
  ///
  /// Note that (1) overrides (2), which overrides (3).

  class Assembler
  {
  public:

    /// Assemble tensor from given form
    static void assemble(GenericTensor& A,
                         const Form& a,
                         bool reset_sparsity=true,
                         bool add_values=false);

    /// Assemble tensor from given form on sub domain
    static void assemble(GenericTensor& A,
                         const Form& a,
                         const SubDomain& sub_domain,
                         bool reset_sparsity=true,
                         bool add_values=false);

    /// Assemble tensor from given form on sub domains
    static void assemble(GenericTensor& A,
                         const Form& a,
                         const MeshFunction<uint>* cell_domains,
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains,
                         bool reset_sparsity=true,
                         bool add_values=false);

  private:

    // Assemble over cells
    static void assemble_cells(GenericTensor& A,
                               const Form& a,
                               UFC& ufc,
                               const MeshFunction<uint>* domains,
                               std::vector<double>* values);

    // Assemble over exterior facets
    static void assemble_exterior_facets(GenericTensor& A,
                                         const Form& a,
                                         UFC& ufc,
                                         const MeshFunction<uint>* domains,
                                         std::vector<double>* values);

    // Assemble over interior facets
    static void assemble_interior_facets(GenericTensor& A,
                                         const Form& a,
                                         UFC& ufc,
                                         const MeshFunction<uint>* domains,
                                         std::vector<double>* values);

  };

}

#endif
