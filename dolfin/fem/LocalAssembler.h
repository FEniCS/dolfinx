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
// Last changed: 2015-09-30

#ifndef __LOCAL_ASSEMBLER_H
#define __LOCAL_ASSEMBLER_H

#include <vector>

#include <dolfin/common/types.h>
#include <Eigen/Dense>

namespace ufc
{
  class cell;
}

namespace dolfin
{

  class Cell;
  class Facet;
  class UFC;
  template<typename T> class MeshFunction;

  /// Assembly of local cell tensors. Used by the adaptivity and
  /// LocalSolver functionality in dolfin. The local assembly
  /// functionality provided here is also wrapped as a free function
  /// assemble_local(form_a, cell) in Python for easier usage. Used
  /// from C++ the versions defined below will be faster than the free
  /// function as less objects need to be created and thrown away

  class LocalAssembler
  {

  public:

    /// Assemble a local tensor on a cell
    static void
      assemble(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>& A,
               UFC& ufc,
               const std::vector<double>& coordinate_dofs,
               ufc::cell& ufc_cell,
               const Cell& cell,
               const MeshFunction<std::size_t>* cell_domains,
               const MeshFunction<std::size_t>* exterior_facet_domains,
               const MeshFunction<std::size_t>* interior_facet_domains);

    /// Worker method called by assemble() to perform assembly of
    /// volume integrals (dx)
    static void assemble_cell(Eigen::Matrix<double, Eigen::Dynamic,
                                            Eigen::Dynamic,
                                            Eigen::RowMajor>& A,
                              UFC& ufc,
                              const std::vector<double>& coordinate_dofs,
                              const ufc::cell& ufc_cell,
                              const Cell& cell,
                              const MeshFunction<std::size_t>* cell_domains);

    /// Worker method called by assemble() for each of the cell's
    /// external facets to perform assembly of external facet
    /// integrals (ds)
    static void
      assemble_exterior_facet(Eigen::Matrix<double, Eigen::Dynamic,
                                            Eigen::Dynamic,
                                            Eigen::RowMajor>& A,
                      UFC& ufc,
                              const std::vector<double>& coordinate_dofs,
                              const ufc::cell& ufc_cell,
                              const Cell& cell,
                              const Facet& facet,
                              const std::size_t local_facet,
                              const MeshFunction<std::size_t>* exterior_facet_domains);

    /// Worker method called by assemble() for each of the cell's
    /// internal facets to perform assembly of internal facet
    /// integrals (dS)
    static void
      assemble_interior_facet(Eigen::Matrix<double, Eigen::Dynamic,
                                            Eigen::Dynamic,
                                            Eigen::RowMajor>& A,
                              UFC& ufc,
                              const std::vector<double>& coordinate_dofs,
                              const ufc::cell& ufc_cell,
                              const Cell& cell,
                              const Facet& facet,
                              const std::size_t local_facet,
                              const MeshFunction<std::size_t>* interior_facet_domains,
                              const MeshFunction<std::size_t>* cell_domains);
  };

}

#endif
