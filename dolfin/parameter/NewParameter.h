// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-05-08
// Last changed: 2009-05-08

#ifndef __NEWPARAMETER_H
#define __NEWPARAMETER_H

#include <string>

namespace dolfin
{

  /// Base class for parameters.

  class NewParameter
  {
  public:

    /// Create parameter for given key
    NewParameter(std::string key);

    /// Destructor
    ~NewParameter();

    /// Return parameter key
    std::string key() const;

    /// Set range for int-valued parameter
    virtual void set_range(int min_value, int max_value);

    /// Set range for double-valued parameter
    virtual void set_range(double min_value, double max_value);

    /// Assignment from int
    virtual const NewParameter& operator= (int value);

    /// Assignment from double
    virtual const NewParameter& operator= (double value);

    /// Cast parameter to int
    virtual operator int() const;

    /// Cast parameter to double
    virtual operator double() const;

    /// Return name of value type
    virtual std::string type_str() const = 0;

    /// Return short string description
    virtual std::string str() const = 0;

  private:

    // Parameter key
    std::string _key;

  };

  /// Parameter with value type int
  class NewIntParameter : public NewParameter
  {
  public:

    /// Create int-valued parameter
    NewIntParameter(std::string key, int value);

    /// Destructor
    ~NewIntParameter();

    /// Set range
    void set_range(int min_value, int max_value);

    /// Assignment
    const NewIntParameter& operator= (int value);

    /// Cast parameter to int
    operator int() const;

    /// Return name of value type
    std::string type_str() const;

    /// Return short string description
    std::string str() const;

  private:

    /// Parameter value
    int _value;

    /// Parameter range
    int _min, _max;

  };

  /// Parameter with value type double
  class NewDoubleParameter : public NewParameter
  {
  public:

    /// Create int-valued parameter
    NewDoubleParameter(std::string key, double value);

    /// Destructor
    ~NewDoubleParameter();

    /// Set range
    void set_range(double min_value, double max_value);

    /// Assignment
    const NewDoubleParameter& operator= (double value);

    /// Cast parameter to int
    operator double() const;

    /// Return name of value type
    std::string type_str() const;

    /// Return short string description
    std::string str() const;

  private:

    /// Parameter value
    double _value;

    /// Parameter range
    double _min, _max;

  };

}

#endif
