// Copyright (C) 2013 Kent-Andre Mardal, Mikael Mortensen, Johan Hake
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
// First added:  2013-04-02

#ifndef __PROBE_H
#define __PROBE_H

#include <vector>
#include <ufc.h>
#include <boost/scoped_ptr.hpp>

namespace dolfin
{
  class Cell;
  class FiniteElement;
  class Function;
  class FunctionSpace;
  template<typename T> class Array;

  class Probe
  {

  public:

    /// Constructor
    Probe(const Array<double>& x, const FunctionSpace& V);

    void eval(const Function& u);

    /// Return probe values for chosen value_size
    std::vector<double> get_probe(std::size_t i) const;

    std::size_t value_size() const;

    std::size_t number_of_evaluations() const;

    /// Return coordinates of probe
    std::vector<double> coordinates() const;

    /// Remove one instance of the probe
    void erase(std::size_t i);

    /// Reset probe by removing all values
    void clear();

  private:

    std::vector<std::vector<double> > _basis_matrix;

    std::vector<double> _coefficients;

    double _x[3];

    std::shared_ptr<const FiniteElement> _element;

    boost::scoped_ptr<Cell> dolfin_cell;

    ufc::cell ufc_cell;

    std::vector<double> _coordinate_dofs;

    std::size_t value_size_loc;

    std::vector<std::vector<double> > _probes;

  };
}

#endif
