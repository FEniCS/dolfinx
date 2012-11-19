// Copyright (C) 2011 Marie E. Rognes
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
// First added:  2011-01-04
// Last changed: 2011-01-13

#ifndef __LOCAL_ASSEMBLER_H
#define __LOCAL_ASSEMBLER_H

#include <vector>

#include <dolfin/common/types.h>

namespace arma
{
  template<typename T> class Mat;
}

namespace dolfin
{

  class Cell;
  class Facet;
  class UFC;
  template<typename T> class MeshFunction;

  ///
  class LocalAssembler
  {

  public:

    ///
    static void assemble(arma::Mat<double>& A,
                         UFC& ufc,
                         const Cell& cell,
                         const MeshFunction<std::size_t>* cell_domains,
                         const MeshFunction<std::size_t>* exterior_facet_domains,
                         const MeshFunction<std::size_t>* interior_facet_domains);

    ///
    static void assemble_cell(arma::Mat<double>& A,
                              UFC& ufc,
                              const Cell& cell,
                              const MeshFunction<std::size_t>* domains);

    ///
    static void assemble_exterior_facet(arma::Mat<double>& A,
                                        UFC& ufc,
                                        const Cell& cell,
                                        const Facet& facet,
                                        const uint local_facet,
                                        const MeshFunction<std::size_t>* domains);

    ///
    static void assemble_interior_facet(arma::Mat<double>& A,
                                        UFC& ufc,
                                        const Cell& cell,
                                        const Facet& facet,
                                        const uint local_facet,
                                        const MeshFunction<std::size_t>* domains);
  };

}

#endif
