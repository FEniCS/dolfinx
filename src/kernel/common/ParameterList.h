// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __PARAMETER_LIST_HH
#define __PARAMETER_LIST_HH

#include <stdarg.h>

#include <dolfin/constants.h>
#include "Parameter.h"
#include "Function.h"

#define KW_FILE_FORMAT_DX    1
#define KW_FILE_FORMAT_INP   2

///
class ParameterList{
public:
  
  ParameterList();
  ~ParameterList();

  /// Initialize and check all parameters
  virtual void Initialize() = 0;
  
  /// Add a parameter
  void Add(const char *identifier, Type type, ...);

  /// Add a function
  void AddFunction(const char *identifier);
  
  /// Set the value of a parameter
  void SetByArgumentList(const char *identifier, va_list aptr);
  void Set(const char *identifier, ...);

  /// Set a function
  void SetFunction(const char *identifier, FunctionPointer f);
  
  /// Get the value of a parameter
  void GetByArgumentList(const char *identifier, va_list aptr);
  void Get(const char *identifier, ...);

  // Get a function
  FunctionPointer GetFunction(const char *identifier);
  
  /// Save all parameters to the default file
  void Save();

  /// Save all parameters to the given file
  void Save(const char *filename);

  /// Load all parameters from the default file
  void Load();

  /// Load all parameters from the given file
  void Load(const char *filename);

  /// Check if the parameter has been changed
  bool Changed(const char *identifier);
  
private:

  int  ParameterIndex(const char *identifier);
  int  FunctionIndex(const char *identifier);
  void Realloc();
  void ReallocFunctions();

  // A list of all the parameters
  Parameter *parameters;

  // A list of all the functions
  Function *functions;

  // Size of list
  int alloc_size;
  int alloc_size_functions;

  // Position for next item
  int current;
  int current_function;
  
};

#endif
