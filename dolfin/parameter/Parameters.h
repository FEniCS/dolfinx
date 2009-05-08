// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-05-08
// Last changed: 2009-05-08

#ifndef __NEWPARAMETERS_H
#define __NEWPARAMETERS_H

#include <map>
#include "NewParameter.h"

namespace dolfin
{

  /// This class stores a database of parameters. Each parameter is
  /// identified by a unique string (the key) and a value of some
  /// given value type.

  class Parameters
  {
  public:

    /// Create empty parameter database
    Parameters(std::string name);

    /// Destructor
    ~Parameters();

    /// Return name of parameter database
    std::string name() const;

    /// Add int-valued parameter
    void add(std::string key, int value);    

    /// Add int-valued parameter with given range
    void add(std::string key, int value, int min_value, int max_value);

    /// Add double-valued parameter
    void add(std::string key, double value);

    /// Add double-valued parameter with given range
    void add(std::string key, double value, double min_value, double max_value);
    
    /// Return parameter for given key
    NewParameter& operator[] (std::string key);

    /// Return parameter for given key (const)
    const NewParameter& operator[] (std::string key) const;

    /// Return short string description
    std::string str() const;

    /// Print parameter data
    void print() const;

  private:

    // Return pointer to parameter for given key and 0 if not found
    NewParameter* find(std::string key) const;

    // Name of database
    std::string _name;

    // Map from parameter key to parameter
    std::map<std::string, NewParameter*> _parameters;

  };

}

#endif
