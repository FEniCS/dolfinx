// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DISPLAY_HH
#define __DISPLAY_HH

#include <stdio.h>
#include <dolfin/constants.h>

/// This class takes care of writing messages in various formats,
/// including progress of the solution and error messages.
//
//  About debug level:
//
//    level = 1   : things that probably everyone is interested in
//    level = 2-9 : in between
//    level = 10  : stuff that is for low-level debug info

enum Type { type_none, type_real, type_double, type_int, type_bool, type_string };

class Display{

public:

  Display(int debug_level){
	 this->debug_level = debug_level;
  };
  
  virtual ~Display(){};
  
  /// Display current status
  virtual void Status(int level, const char *format, ...) = 0;

  /// Display a message
  virtual void Message(int level, const char *format, ...) = 0;
  
  /// Display progress
  virtual void Progress(int level, double progress, const char *format, ...) = 0;
  
  /// Display regress (progress backwards)
  virtual void Regress(int level, double progress, double maximum, const char *format, ...) = 0;

  /// Display a value
  virtual void Value(const char *name, Type type, ...) = 0;
  
  /// Display a warning
  virtual void Warning(const char *format, ...) = 0;
  
  /// Display an error message
  virtual void Error(const char *format, ...) = 0;
  
  /// Display an "internal error" message
  virtual void InternalError(const char *function, const char *format, ...) = 0;
  
protected:

  int StringLength(const char *string);
  
  FILE *logfile;

  int debug_level;
  
};

extern Display *display;

#endif
