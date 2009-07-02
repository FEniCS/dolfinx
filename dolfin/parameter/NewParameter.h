// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-05-08
// Last changed: 2009-06-30

#ifndef __NEWPARAMETER_H
#define __NEWPARAMETER_H

#include <set>
#include <string>
#include <dolfin/common/types.h>

namespace dolfin
{

  /// Base class for parameters.

  class NewParameter
  {
  public:

    /// Create parameter for given key
    NewParameter(std::string key);

    /// Destructor
    virtual ~NewParameter();

    /// Return parameter key
    std::string key() const;

    /// Return parameter description
    std::string description() const;

    /// Return access count (number of times parameter has been accessed)
    uint access_count() const;

    /// Return change count (number of times parameter has been changed)
    uint change_count() const;

    /// Set range for int-valued parameter
    virtual void set_range(int min_value, int max_value);

    /// Set range for double-valued parameter
    virtual void set_range(double min_value, double max_value);

    /// Set range for string-valued parameter
    virtual void set_range(const std::set<std::string>& range);

    /// Assignment from int
    virtual const NewParameter& operator= (int value);

    /// Assignment from double
    virtual const NewParameter& operator= (double value);

    /// Assignment from string
    virtual const NewParameter& operator= (std::string value);

    /// Assignment from string
    virtual const NewParameter& operator= (const char* value);

    /// Assignment from bool
    virtual const NewParameter& operator= (bool value);

    /// Cast parameter to int
    virtual operator int() const;

    /// Cast parameter to uint
    virtual operator dolfin::uint() const;

    /// Cast parameter to double
    virtual operator double() const;

    /// Cast parameter to string
    virtual operator std::string() const;

    /// Cast parameter to bool
    virtual operator bool() const;

    /// Return value type string
    virtual std::string type_str() const = 0;

    /// Return value string
    virtual std::string value_str() const = 0;

    /// Return range string
    virtual std::string range_str() const = 0;

    /// Return short string description
    virtual std::string str() const = 0;
    
    // Check that key name is allowed
    static void check_key(std::string key);

  protected:

    // Access count
    mutable uint _access_count;

    // Change count
    uint _change_count;

  private:

    // Parameter key
    std::string _key;

    // Parameter description
    std::string _description;

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

    /// Cast parameter to uint
    operator dolfin::uint() const;

    /// Return value type string
    std::string type_str() const;

    /// Return value string
    std::string value_str() const;

    /// Return range string
    std::string range_str() const;

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

    /// Create double-valued parameter
    NewDoubleParameter(std::string key, double value);

    /// Destructor
    ~NewDoubleParameter();

    /// Set range
    void set_range(double min_value, double max_value);

    /// Assignment
    const NewDoubleParameter& operator= (double value);

    /// Cast parameter to double
    operator double() const;

    /// Return value type string
    std::string type_str() const;

    /// Return value string
    std::string value_str() const;

    /// Return range string
    std::string range_str() const;

    /// Return short string description
    std::string str() const;

  private:

    /// Parameter value
    double _value;

    /// Parameter range
    double _min, _max;

  };

  /// Parameter with value type string
  class NewStringParameter : public NewParameter
  {
  public:

    /// Create string-valued parameter
    NewStringParameter(std::string key, std::string value);

    /// Destructor
    ~NewStringParameter();

    /// Set range
    void set_range(const std::set<std::string>& range);

    /// Assignment
    const NewStringParameter& operator= (std::string value);

    /// Assignment
    const NewStringParameter& operator= (const char* value);

    /// Cast parameter to string
    operator std::string() const;

    /// Return value type string
    std::string type_str() const;

    /// Return value string
    std::string value_str() const;

    /// Return range string
    std::string range_str() const;

    /// Return short string description
    std::string str() const;

  private:

    /// Parameter value
    std::string _value;

    /// Parameter range
    std::set<std::string> _range;

  };

  /// Parameter with value type bool
  class NewBoolParameter : public NewParameter
  {
  public:

    /// Create bool-valued parameter
    NewBoolParameter(std::string key, bool value);

    /// Destructor
    ~NewBoolParameter();

    /// Assignment
    const NewBoolParameter& operator= (bool value);

    /// Cast parameter to bool
    operator bool() const;

    /// Return value type string
    std::string type_str() const;

    /// Return value string
    std::string value_str() const;

    /// Return range string
    std::string range_str() const;

    /// Return short string description
    std::string str() const;

  private:

    /// Parameter value
    bool _value;

  };

}

#endif
