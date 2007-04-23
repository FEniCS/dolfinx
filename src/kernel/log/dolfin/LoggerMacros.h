// Copyright (C) 2003-2005 Anders Logg and Jim Tilander.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-03-13
// Last changed: 2005-09-15

#ifndef __LOGGER_MACROS_H
#define __LOGGER_MACROS_H

#include <stdarg.h>
#include <dolfin/LogManager.h>

// Info (does not need to be a macro)
namespace dolfin { void dolfin_info(const char *msg, ...); }
namespace dolfin { void dolfin_info_aptr(const char *msg, va_list aptr); }

// Update (force refresh of curses interface)
namespace dolfin { void dolfin_update(); }

// Stop program
namespace dolfin { void dolfin_quit(); }

// Check if the computation is still running
namespace dolfin { bool dolfin_finished(); }

// Raise a segmentation fault, useful for debugging
namespace dolfin { void dolfin_segfault(); }

// Task notification
namespace dolfin { void dolfin_begin(); }
namespace dolfin { void dolfin_begin(const char* msg, ...); }
namespace dolfin { void dolfin_end(); }
namespace dolfin { void dolfin_end(const char* msg, ...); }

// Specify output type ("plain text", "curses", or "silent")
namespace dolfin { void dolfin_output(const char* destination); }

// Switch logging on or off
namespace dolfin { void dolfin_log(bool state); }

// Debug macros (with varying number of arguments)

#define dolfin_debug(msg)                  do { dolfin::LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg); } while( false )
#define dolfin_debug1(msg, a0)             do { dolfin::LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while( false )
#define dolfin_debug2(msg, a0, a1)         do { dolfin::LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while( false )
#define dolfin_debug3(msg, a0, a1, a2)     do { dolfin::LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while( false )
#define dolfin_debug4(msg, a0, a1, a2, a3) do { dolfin::LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2, a3); } while( false )

// Warning macros (with varying number of arguments)

#define dolfin_warning(msg)                  do { dolfin::LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg); } while( false )
#define dolfin_warning1(msg, a0)             do { dolfin::LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while( false )
#define dolfin_warning2(msg, a0, a1)         do { dolfin::LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while( false )
#define dolfin_warning3(msg, a0, a1, a2)     do { dolfin::LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while( false )
#define dolfin_warning4(msg, a0, a1, a2, a3) do { dolfin::LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2, a3); } while( false )

// Error macros (with varying number of arguments)

#define dolfin_error(msg)                  do { dolfin::LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg); } while( false )
#define dolfin_error1(msg, a0)             do { dolfin::LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while( false )
#define dolfin_error2(msg, a0, a1)         do { dolfin::LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while( false )
#define dolfin_error3(msg, a0, a1, a2)     do { dolfin::LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while( false )
#define dolfin_error4(msg, a0, a1, a2, a3) do { dolfin::LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2, a3); } while( false )

// Assertion, turned off if DEBUG is not defined

#ifdef DEBUG
#define dolfin_assert(check) do { if ( !(check) ) dolfin::LogManager::log.dassert(__FILE__, __LINE__, __FUNCTION__, "(" #check ")"); } while ( false )
#else
#define dolfin_assert(check)
#endif

#endif
