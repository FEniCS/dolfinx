#ifndef __PROBE_H
#define __PROBE_H

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>

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



namespace dolfin
{
  class Function;  
  class FunctionSpace;    
  template<typename T> class Array;

  class Probe
  {
    
  public:
      
    Probe(const Array<double>& x, const FunctionSpace& V);

    void eval(const Function& u);
    
    std::vector<double> get_probe(std::size_t i);
    
    std::size_t value_size();
    
    std::size_t number_of_evaluations();
    
    std::vector<double> coordinates();
    
    void erase(std::size_t i);
    
    void clear();
    
  private:
      
    std::vector<std::vector<double> > basis_matrix;
    
    std::vector<double> coefficients;
    
    double _x[3];
    
    boost::shared_ptr<const FiniteElement> _element;
    
    Cell* dolfin_cell;
    
    UFCCell* ufc_cell;
    
    std::size_t value_size_loc;    
    
    std::vector<std::vector<double> > _probes;

  };
}

#endif
