// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TERMINAL_HH
#define __TERMINAL_HH

#include <dolfin/Display.hh>

class Terminal : public Display {

public:

  Terminal(int debug_level) : Display(debug_level) {};
  
  ~Terminal() {};
  
  /// Display current status
  void Status(int level, const char *format, ...);

  /// Display a message: should work like printf
  void Message(int level, const char *format, ...);
  
  /// Display progress
  void Progress(int level, double progress, const char *format, ...);

  /// Display regress (progress backwards)
  void Regress(int level, double progress, double maximum, const char *format, ...);
  
  /// Display a value
  void Value(const char *name, Type type, ...);
  
  /// Display a warning
  void Warning(const char *format, ...);
  
  /// Display an error message
  void Error(const char *format, ...);
  
  /// Display an "internal error" message
  void InternalError(const char *function, const char *format, ...);
	 
private:

  
};

#endif
