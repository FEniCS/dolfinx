// Copyright (C) 2007-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007-2008.
// Modified by Ola Skavhaug, 2008.
//
// First added:  2007-01-17
// Last changed: 2010-11-18

#ifndef __OPENMP_ASSEMBLER_H
#define __OPENMP_ASSEMBLER_H

#ifdef HAS_OPENMP

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
  /// The MeshFunction arguments can be used to specify assembly over
  /// subdomains of the mesh cells, exterior facets or interior
  /// facets. Either a null pointer or an empty MeshFunction may be
  /// used to specify that the tensor should be assembled over the
  /// entire set of cells or facets.

  class OpenMpAssembler
  {
  public:

    /// Assemble tensor from given form
    static void assemble(GenericTensor& A,
                         const Form& a,
                         bool reset_sparsity=true,
                         bool add_values=false);

  private:

    friend class Assembler;

    /// Assemble tensor from given form on sub domains
    static void assemble(GenericTensor& A,
                         const Form& a,
                         const MeshFunction<uint>* cell_domains,
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains,
                         bool reset_sparsity=true,
                         bool add_values=false);

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
#endif
