// Copyright (C) 2003 Jim Tilander.
// Licensed under the GNU GPL Version 2.
//
// Modified for DOLFIN by Anders Logg.

#ifndef __LOGGER_MACROS_H
#define __LOGGER_MACROS_H

#include <dolfin/LogManager.h>

// Info (does not need to be a macro)
namespace dolfin { void dolfin_info(const char *msg, ...); }

// Update (force refresh of curses interface)
namespace dolfin { void dolfin_update(); }

// Stop program
namespace dolfin { void dolfin_quit(); }

// Check if the computation is still running
namespace dolfin { bool dolfin_finished(); }

// Raise a segmentation fault, useful for debugging
namespace dolfin { void dolfin_segfault(); }

// Debug macros (with varying number of arguments)

#define dolfin_debug(msg) \
do { dolfin::LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg); } while( false )
#define dolfin_debug1(msg, a0) \
do { dolfin::LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while( false )
#define dolfin_debug2(msg, a0, a1) \
do { dolfin::LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while( false )
#define dolfin_debug3(msg, a0, a1, a2) \
do { dolfin::LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while( false )
#define dolfin_debug4(msg, a0, a1, a3) \
do { dolfin::LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2, a3); } while( false )

// Warning macros (with varying number of arguments)

#define dolfin_warning(msg) \
do { dolfin::LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg); } while( false )
#define dolfin_warning1(msg, a0) \
do { dolfin::LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while( false )
#define dolfin_warning2(msg, a0, a1) \
do { dolfin::LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while( false )
#define dolfin_warning3(msg, a0, a1, a2) \
do { dolfin::LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while( false )
#define dolfin_warning4(msg, a0, a1, a3) \
do { dolfin::LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2, a3); } while( false )

// Error macros (with varying number of arguments)

#define dolfin_error(msg) \
do { dolfin::LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg); } while( false )
#define dolfin_error1(msg, a0) \
do { dolfin::LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while( false )
#define dolfin_error2(msg, a0, a1) \
do { dolfin::LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while( false )
#define dolfin_error3(msg, a0, a1, a2) \
do { dolfin::LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while( false )
#define dolfin_error4(msg, a0, a1, a3) \
do { dolfin::LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2, a3); } while( false )

// Assertion, turned off if DEBUG is not defined

#ifdef DEBUG
#define dolfin_assert(check) \
do { if ( !(check) ) dolfin::LogManager::log.dassert(__FILE__, __LINE__, __FUNCTION__, "(" #check ")"); } while ( false )
#else
#define dolfin_assert(check)
#endif

// Macros for task notification

#define dolfin_start do { dolfin::LogManager::log.start(); } while ( false )
#define dolfin_end   do { dolfin::LogManager::log.end();   } while ( false )

#endif
