// Copyright (C) 2003-2007 Anders Logg and Jim Tilander.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// 
// First added:  2003-03-13
// Last changed: 2007-05-11


#ifndef __LOG_H
#define __LOG_H

#include <stdarg.h>

namespace dolfin
{
  
  // Print message
  void dolfin_info(const char *msg, ...);

  // Print message
  void dolfin_info(int debug_level, const char* msg, ...);

  // Print message
  void dolfin_info_aptr(const char *msg, va_list aptr);

  // Begin task (increase indentation level)
  void dolfin_begin();

  // Begin task (increase indentation level)
  void dolfin_begin(const char* msg, ...);

  // Begin task (increase indentation level)
  void dolfin_begin(int debug_level, const char* msg, ...);

  // End task (decrease indentation level)
  void dolfin_end();

  // Set output destination ("terminal", or "silent")
  void dolfin_log(const char* destination);

  // Set debug level
  void dolfin_log(int debug_level);

  // Switch logging on or off
  void dolfin_log(bool state);

  void debug(const char* file, unsigned long line, const char* function, const char* format, ...);
  void warning(const char* file, unsigned long line, const char* function, const char* format, ...);
  void error(const char* file, unsigned long line, const char* function, const char* format, ...);
  void dassert  (const char* file, unsigned long line, const char* function, const char* format, ...);

}

// Debug macros (with varying number of arguments)
//#define dolfin_debug(msg)              do { dolfin_debug(__FILE__, __LINE__, __FUNCTION__, msg); } while (false)
#define dolfin_debug(msg)              do { dolfin::debug(__FILE__, __LINE__, __FUNCTION__, msg); } while (false)
#define dolfin_debug1(msg, a0)         do { dolfin::debug(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while (false)
#define dolfin_debug2(msg, a0, a1)     do { dolfin::debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while (false)
#define dolfin_debug3(msg, a0, a1, a2) do { dolfin::debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while (false)

// Warning macros (with varying number of arguments)
#define dolfin_warning(msg)              do { dolfin::warning(__FILE__, __LINE__, __FUNCTION__, msg); } while (false)
#define dolfin_warning1(msg, a0)         do { dolfin::warning(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while (false)
#define dolfin_warning2(msg, a0, a1)     do { dolfin::warning(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while (false)
#define dolfin_warning3(msg, a0, a1, a2) do { dolfin::warning(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while (false)

// Error macros (with varying number of arguments)
#define dolfin_error(msg)              do { dolfin::error(__FILE__, __LINE__, __FUNCTION__, msg); } while (false)
#define dolfin_error1(msg, a0)         do { dolfin::error(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while (false)
#define dolfin_error2(msg, a0, a1)     do { dolfin::error(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while (false)
#define dolfin_error3(msg, a0, a1, a2) do { dolfin::error(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while (false)

// Assertion, only active if DEBUG is defined
#ifdef DEBUG
#define dolfin_assert(check) do { if ( !(check) ) dolfin::dassert(__FILE__, __LINE__, __FUNCTION__, "(" #check ")"); } while (false)
#else
#define dolfin_assert(check)
#endif

#endif
