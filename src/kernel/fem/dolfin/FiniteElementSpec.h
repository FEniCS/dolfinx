// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-19
// Last changed: 2006-02-19

#ifndef __FINITE_ELEMENT_SPEC_H
#define __FINITE_ELEMENT_SPEC_H

#include <string>

#include <dolfin/dolfin_log.h>

namespace dolfin
{
  
  class FiniteElementSpec
  {
  public:
    
    /// Empty constructor
    FiniteElementSpec();

    /// Create given finite element specification
    FiniteElementSpec(std::string type, std::string shape, uint degree, uint vectordim = 0);

    /// Initialize given finite element specification
    void init(std::string type, std::string shape, uint degree, uint vectordim = 0);

    /// Return type of finite element
    std::string type() const;

    /// Return shape of finite element
    std::string shape() const;

    /// Return degree of finite element
    uint degree() const;

    /// Return vector dimension of finite element
    uint vectordim() const;

    /// Return string representation
    std::string repr() const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const FiniteElementSpec& spec);

  private:

    std::string _type;
    std::string _shape;
    uint _degree;
    uint _vectordim;

  };

}

#endif
