// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Based on a prototype implementation by Didem Unat.
//
// First added:  2010-11-04
// Last changed: 2010-11-08

#ifndef __MULTICORE_ASSEMBLER_H
#define __MULTICORE_ASSEMBLER_H

namespace dolfin
{

  /// This class implements shared-memory parallel assembly based on
  /// threads. It may be used directly, but it will be automatically
  /// invoked by the normal DOLFIN assembler whenever the global
  /// parameter "num_threads" is set to a value larger than 1.

  class MulticoreAssembler
  {
  public:

    /// Assemble tensor from given form on sub domains
    static void assemble(GenericTensor& A,
                         const Form& a,
                         const MeshFunction<uint>* cell_domains,
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains,
                         bool reset_sparsity,
                         bool add_values,
                         uint num_threads);

  private:

    // Multi-thread assembly (create and join threads)
    static void assemble_threads(GenericTensor* A,
                                 const Form* a,
                                 UFC* ufc,
                                 uint num_threads,
                                 const MeshFunction<uint>* cell_domains,
                                 const MeshFunction<uint>* exterior_facet_domains,
                                 const MeshFunction<uint>* interior_facet_domains);

    // Single-thread assembly (called by each thread)
    static void assemble_thread(GenericTensor* A,
                                const Form* a,
                                UFC* ufc,
                                uint thread_id,
                                uint num_threads,
                                const MeshFunction<uint>* cell_domains,
                                const MeshFunction<uint>* exterior_facet_domains,
                                const MeshFunction<uint>* interior_facet_domains);

    // Assemble over cells
    static void assemble_cells(GenericTensor& A,
                               const Form& a,
                               UFC& ufc,
                               const std::pair<uint, uint>& range,
                               const MeshFunction<uint>* domains,
                               std::vector<double>* values);

    // Assemble over exterior facets
    static void assemble_exterior_facets(GenericTensor& A,
                                         const Form& a,
                                         UFC& ufc,
                                         const std::pair<uint, uint>& range,
                                         const MeshFunction<uint>* domains,
                                         std::vector<double>* values);

    // Assemble over interior facets
    static void assemble_interior_facets(GenericTensor& A,
                                         const Form& a,
                                         UFC& ufc,
                                         const std::pair<uint, uint>& range,
                                         const MeshFunction<uint>* domains,
                                         std::vector<double>* values);

  };

}

#endif
