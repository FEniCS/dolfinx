// Copyright (C) 2015 Tormod Landet
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
// First added:  2015-09-22
//
// This file adds an easy to use wrapper for the LocalAssembler::assemble
// routine that can used from Python

#ifndef __ASSEMBLE_LOCAL_H
#define __ASSEMBLE_LOCAL_H

#include <Eigen/Dense>
#include <vector>

namespace dolfin
{
  class Form;
  class Cell;

  /// Assemble form to local tensor on a cell (Eigen version for pybind11)
  void assemble_local(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& A_e,
                      const Form& a, const Cell& cell);

  /// Assemble form to local tensor on a cell
  /// (Legacy version for SWIG)
  void assemble_local(const Form& a,
                      const Cell& cell,
                      std::vector<double>& tensor);
}

#endif
