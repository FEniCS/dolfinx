// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-10-28
// Last changed: 2008-10-28

#ifndef __COEFFICIENT_H
#define __COEFFICIENT_H

#include <tr1/memory>

namespace dolfin
{

  // Forward declarations
  class Form;
  class Function;
  class FunctionSpace;

  /// This class is used for assignment of functions to the
  /// coefficients of a form, which allows magic like
  ///
  ///   a.f = f
  ///   a.g = g
  ///
  /// which at the same time will attach functions to the
  /// coefficients of a form and set the correct function
  /// spaces (if missing) for the functions.

  class Coefficient
  {
  public:

    /// Constructor
    Coefficient(Form& form);

    /// Destructor
    virtual ~Coefficient();

    /// Assign function to coefficient
    const Coefficient& operator= (Function& v);

    /// Create function space for coefficient
    virtual const FunctionSpace* create_function_space() const = 0;

    /// Return number of coefficient
    virtual uint number() const = 0;

    /// Return name of coefficient
    virtual std::string name() const = 0;

  private:

    // The form
    Form& form;

  };

}

#endif
