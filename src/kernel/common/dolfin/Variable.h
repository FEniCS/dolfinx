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

	 void rename(const std::string name, const std::string label);

	 const std::string& name()  const;
	 const std::string& label() const;

  private:

	 std::string _name;
	 std::string _label;

  };

}

#endif
