// Copyright (C) 2005-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-12-18
// Last changed: 2009-05-06

#ifndef __PARAMETER_VALUE_H
#define __PARAMETER_VALUE_H

#include <string>
#include <dolfin/common/types.h>

namespace dolfin
{

  /// Base class for parameter values
  class ParameterValue
  {
  public:

    /// Constructor
    ParameterValue();

    /// Destructor
    virtual ~ParameterValue();

    /// Set int value
    virtual void set(int value);

    /// Set uint value
    virtual void set(uint value);

    /// Set double value
    virtual void set(double value);

    /// Set bool value
    virtual void set(bool value);

    /// Set string value
    virtual void set(std::string value);

    /// Set int-valued range
    void set_range(int min_value, int max_value);

    /// Set uint-valued range
    void set_range(uint min_value, uint max_value);
    
    /// Set double-valued range
    void set_range(double min_value, double max_value);
    
    /// Set string-valued range (list of values)
    void set_range(const std::vector<std::string>& allowed_values);

    /// Cast to int
    virtual operator int() const;

    /// Cast to uint
    virtual operator uint() const;

    /// Cast to double
    virtual operator double() const;

    /// Cast to bool
    virtual operator bool() const;

    /// Cast to string
    virtual operator std::string() const;

    /// Name of value type
    virtual std::string type_str() const = 0;

  };

  /// int-valued parameter value
  class IntValue : public ParameterValue
  {
  public:

    /// Constructor
    IntValue(int value);

    /// Destructor
    ~IntValue();

    /// Set int value
    void set(int value);

    /// Set uint value
    void set(uint value);

    /// Set int-valued range
    void set_range(int min_value, int max_value);

    /// Set uint-valued range
    void set_range(uint min_value, uint max_value);

    /// Cast to int
    operator int() const;

    /// Cast to uint
    operator uint() const;

    /// Name of value type
    std::string type_str() const;

  private:

    int value;
    int min_value;
    int max_value;

  };

  /// double-valued parameter value
  class DoubleValue : public ParameterValue
  {
  public:

    /// Constructor
    DoubleValue(double value);

    /// Destructor
    ~DoubleValue();

    /// Set real value
    void set(double value);

    /// Set double-valued range
    void set_range(double min_value, double max_value);
    
    /// Cast to double
    operator double() const;

    /// Name of value type
    std::string type_str() const;

  private:

    double value;
    double min_value;
    double max_value;

  };

  /// bool-valued parameter value
  class BoolValue : public ParameterValue
  {
  public:

    /// Constructor
    BoolValue(bool value);

    /// Destructor
    ~BoolValue();

    /// Set bool value
    void set(bool value);

    /// Cast to int
    operator bool() const;

    /// Name of value type
    std::string type_str() const;

  private:

    bool value;

  };

  /// string-valued parameter value
  class StringValue : public ParameterValue
  {
  public:

    /// Constructor
    StringValue(std::string value);

    /// Destructor
    ~StringValue();

    /// Set string value
    void set(std::string value);

    /// Set string-valued range (list of values)
    void set_range(const std::vector<std::string>& allowed_values);

    /// Cast to string
    operator std::string() const;

    /// Name of value type
    std::string type_str() const;

  private:

    std::string value;
    std::vector<std::string> allowed_values;

  };

}

#endif
