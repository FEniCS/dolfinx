// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PARAMETER_HH
#define __PARAMETER_HH

#include <iostream>
#include <string.h>

#include <dolfin/constants.h>
#include <dolfin/function.h>
#include "utils.h"

#define PARAMETER_IDENTIFIER_LENGTH 128

namespace dolfin {

  // A small class for internal use in Settings
  class Parameter{
	 
  public:

	 enum Type { REAL, INT, BOOL, STRING, FUNCTION, NONE };
	 
	 Parameter();
	 Parameter(Type type, const char *identifier, va_list aptr);

	 ~Parameter();

	 void clear();

	 void set(Type type, const char *identifier, va_list aptr);
	 void set(const char *identifier, va_list aptr);
	 
	 void get(va_list aptr);
	 
	 bool matches(const char *string);
	 bool changed();

	 void operator= (const Parameter &p);
	 
	 // Needed for ShortList
	 void operator= (int zero);
	 bool operator! () const;
	 
  private:

	 // A description of the parameter
	 char identifier[PARAMETER_IDENTIFIER_LENGTH];
	 
	 // Values
	 real      val_real;
	 int       val_int;
	 char     *val_string;
	 function  val_function;
	 
	 // True iff the default value was changed
	 bool _changed;
	 
	 // Type of data
	 Type type;
	 
  };

}

#endif
