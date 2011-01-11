// Copyright (C) 2011 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
//
// First added:  2011-01-04
// Last changed: 2011-01-05

#ifndef __LOCAL_ASSEMBLER_H
#define __LOCAL_ASSEMBLER_H

#include <vector>
#include <armadillo>

#include <dolfin/common/types.h>

namespace dolfin
{

  class UFC;
  class Cell;

  ///
  class LocalAssembler
  {

  public:

    ///
    static void assemble_cell(arma::mat& A,
                              const uint N,
                              UFC& ufc,
                              const Cell& cell,
                              std::vector<uint> exterior_facets,
                              std::vector<uint> interior_facets);

    ///
    static void assemble_cell(arma::vec& b,
                              const uint N,
                              UFC& ufc,
                              const Cell& cell,
                              std::vector<uint> exterior_facets,
                              std::vector<uint> interior_facets);
  };

}

#endif
