// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-02-26
// Last changed: 2006-10-09

#ifndef __VARIABLE_H
#define __VARIABLE_H

#include <string>

namespace dolfin
{
  
  class Variable
  {
  public:
    
    Variable();
    Variable(const std::string name, const std::string label);
    Variable(const Variable& variable);
    
    void rename(const std::string name, const std::string label);
    
    const std::string& name()  const;
    const std::string& label() const;

  private:
    
    std::string _name;
    std::string _label;
    
  };
  
}

#endif
