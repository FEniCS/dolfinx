// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CURSES_LOGGER_H
#define __CURSES_LOGGER_H

#include <curses.h>

#include <dolfin/constants.h>
#include <dolfin/Buffer.h>
#include <dolfin/GenericLogger.h>

namespace dolfin {
  
  class CursesLogger : public GenericLogger {
  public:
    
    CursesLogger();
    ~CursesLogger();
    
    void info    (const char* msg);
    void debug   (const char* msg, const char* location);
    void warning (const char* msg, const char* location);
    void error   (const char* msg, const char* location);
    void progress(const char* title, const char* label, real p);

    void update();
    
  private:

    enum State { BUFFER, ABOUT, HELP };

    State state;    // State (what to display)
    
    WINDOW *win;    // Pointer to the terminal
    
    int lines;      // Number of lines
    int cols;       // Number of columns
    
    int offset;     // Start position for buffer
    Buffer buffer;  // Buffer
    char*  guiinfo; // Message from the curses interface (not program message)

    bool finished;  // True if finished
    
    void setSignals();
    
    void setInfo(const char* msg);
    
    void clearLines();
    void clearLine(int line, int col);

    bool getYesNo();
    void getAnyKey();
    
    void drawTitle();
    void drawBuffer();
    void drawAbout();
    void drawHelp();
    void drawInfo();
    void drawCommands();
    
    void redraw();
    
  };
  
}

#endif
