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
    Parameters(std::string key);

    /// Destructor
    ~Parameters();

    /// Copy constructor
    Parameters(const Parameters& parameters);

    /// Return database key
    std::string key() const;
    
    /// Clear database
    void clear();

    /// Add int-valued parameter
    void add(std::string key, int value);    

    /// Add int-valued parameter with given range
    void add(std::string key, int value, int min_value, int max_value);

    /// Add double-valued parameter
    void add(std::string key, double value);

    /// Add double-valued parameter with given range
    void add(std::string key, double value, double min_value, double max_value);

    /// Add nested parameter database
    void add(const Parameters& parameters);

    /// Return parameter for given key
    NewParameter& operator() (std::string key);

    /// Return parameter for given key (const)
    const NewParameter& operator() (std::string key) const;

    /// Return nested parameter database for given key
    Parameters& operator[] (std::string key);

    /// Return nested parameter database for given key (const)
    const Parameters& operator[] (std::string key) const;

    /// Assignment operator
    const Parameters& operator= (const Parameters& parameters);

    /// Return short string description
    std::string str() const;

    /// Print parameter data
    void print() const;

  private:

    // Return pointer to parameter for given key and 0 if not found
    NewParameter* find_parameter(std::string key) const;

    // Return pointer to database for given key and 0 if not found
    Parameters* find_database(std::string key) const;

    // Database key
    std::string _key;

    // Map from key to parameter
    std::map<std::string, NewParameter*> _parameters;

    // Map from key to database
    std::map<std::string, Parameters*> _databases;

  };

}

#endif
