// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __VARIABLE_H
#define __VARIABLE_H

#include <string>

namespace dolfin {

  class Variable {
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
