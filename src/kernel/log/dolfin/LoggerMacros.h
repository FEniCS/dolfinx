// Copyright (C) 2003 Jim Tilander.
// Licensed under the GNU GPL Version 2.
//
// Modified for DOLFIN by Anders Logg.

#ifndef __LOGGER_MACROS_H
#define __LOGGER_MACROS_H

#include <dolfin/LogManager.h>

// Info (does not need to be a macro)

namespace dolfin { void dolfin_info(const char *msg, ...); }

// Debug macros (with varying number of arguments)

#define dolfin_debug(msg) \
do { LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg); } while( false )
#define dolfin_debug1(msg, a0) \
do { LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while( false )
#define dolfin_debug2(msg, a0, a1) \
do { LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while( false )
#define dolfin_debug3(msg, a0, a1, a2) \
do { LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while( false )
#define dolfin_debug4(msg, a0, a1, a3) \
do { LogManager::log.debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2, a3); } while( false )

// Warning macros (with varying number of arguments)

#define dolfin_warning(msg) \
do { LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg); } while( false )
#define dolfin_warning1(msg, a0) \
do { LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while( false )
#define dolfin_warning2(msg, a0, a1) \
do { LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while( false )
#define dolfin_warning3(msg, a0, a1, a2) \
do { LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while( false )
#define dolfin_warning4(msg, a0, a1, a3) \
do { LogManager::log.warning(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2, a3); } while( false )

// Error macros (with varying number of arguments)

#define dolfin_error(msg) \
do { LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg); } while( false )
#define dolfin_error1(msg, a0) \
do { LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while( false )
#define dolfin_error2(msg, a0, a1) \
do { LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while( false )
#define dolfin_error3(msg, a0, a1, a2) \
do { LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while( false )
#define dolfin_error4(msg, a0, a1, a3) \
do { LogManager::log.error(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2, a3); } while( false )

// Macros for task notification

#define dolfin_start do { LogManager::log.start(); } while ( false )
#define dolfin_end   do { LogManager::log.end();   } while ( false )

#endif
