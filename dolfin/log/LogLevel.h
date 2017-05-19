#ifndef __LOGLEVEL_H
#define __LOGLEVEL_H

namespace dolfin
{
  /// These log levels match the levels in the Python 'logging' module
  /// (and adds trace/progress).
  enum LogLevel
  {
    CRITICAL  = 50, // errors that may lead to data corruption and suchlike
    ERROR     = 40, // things that go boom
    WARNING   = 30, // things that may go boom later
    INFO      = 20, // information of general interest
    PROGRESS  = 16, // what's happening (broadly)
    TRACE     = 13, // what's happening (in detail)
    DBG       = 10  // sundry
  };
}

#endif
