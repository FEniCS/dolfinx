// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-05-08
// Last changed: 2009-05-08

#ifndef __NEWPARAMETERS_H
#define __NEWPARAMETERS_H

#include <set>
#include <map>
#include <vector>

namespace dolfin
{

  class NewParameter;

  /// This class stores a database of parameters. Each parameter is
  /// identified by a unique string (the key) and a value of some
  /// given value type.

  class NewParameters
  {
  public:

    /// Create empty parameter database
    NewParameters(std::string key="parameters");

    /// Destructor
    ~NewParameters();

    /// Copy constructor
    NewParameters(const NewParameters& parameters);

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

    /// Add string-valued parameter
    void add(std::string key, std::string value);

    /// Add string-valued parameter with given range
    void add(std::string key, std::string value, std::set<std::string> range);

    /// Add nested parameter database
    void add(const NewParameters& parameters);

    /// Parse parameters from command-line
    void parse(int argc, char* argv[]);

    /// Return parameter for given key
    NewParameter& operator() (std::string key);

    /// Return parameter for given key (const version)
    const NewParameter& operator() (std::string key) const;

    /// Return nested parameter database for given key
    NewParameters& operator[] (std::string key);

    /// Return nested parameter database for given key (const)
    const NewParameters& operator[] (std::string key) const;

    /// Assignment operator
    const NewParameters& operator= (const NewParameters& parameters);

    /// Return informal string representation (pretty-print)
    std::string str() const;

    /// Return a vector of keys to the parameters
    void parameter_keys(std::vector<std::string>& keys) const;

    /// Return a vector of keys to the database
    void database_keys(std::vector<std::string>& keys) const;

  private:

    // Return pointer to parameter for given key and 0 if not found
    NewParameter* find_parameter(std::string key) const;

    // Return pointer to database for given key and 0 if not found
    NewParameters* find_database(std::string key) const;

    // Database key
    std::string _key;

    // Map from key to parameter
    std::map<std::string, NewParameter*> _parameters;

    // Map from key to database
    std::map<std::string, NewParameters*> _databases;

  };

}

#endif
