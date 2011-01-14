// Copyright (C) 2011 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
//
// First added:  2011-01-04
// Last changed: 2011-01-13

#ifndef __LOCAL_ASSEMBLER_H
#define __LOCAL_ASSEMBLER_H

#include <vector>
#include <armadillo>

#include <dolfin/common/types.h>

namespace dolfin
{

  class UFC;
  class Cell;
  template<class T> class MeshFunction;

  ///
  class LocalAssembler
  {

  public:

    ///
    static void assemble(arma::mat& A,
                         UFC& ufc,
                         const Cell& cell,
                         const MeshFunction<uint>* cell_domains,
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains);

    ///
    static void assemble_cell(arma::mat& A,
                              UFC& ufc,
                              const Cell& cell,
                              const MeshFunction<uint>* domains);

    ///
    static void assemble_exterior_facet(arma::mat& A,
                                        UFC& ufc,
                                        const Cell& cell,
                                        const Facet& facet,
                                        const uint local_facet,
                                        const MeshFunction<uint>* domains);

    ///
    static void assemble_interior_facet(arma::mat& A,
                                        UFC& ufc,
                                        const Cell& cell,
                                        const Facet& facet,
                                        const uint local_facet,
                                        const MeshFunction<uint>* domains);
  };

}

#endif
