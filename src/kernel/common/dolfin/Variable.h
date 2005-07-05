// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-02-26
// Last changed: 2005

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

    void operator++();
    int number() const;
    
  private:
    
    std::string _name;
    std::string _label;

    // Number of times variable has been saved to file
    int _number;
    
  };
  
}

#endif
