// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <sys/types.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>

#include <dolfin/constants.h>
#include <dolfin/utils.h>
#include <dolfin/timeinfo.h>
#include <dolfin/LoggerMacros.h>
#include <dolfin/CursesLogger.h>

using namespace dolfin;

// Signal handlers
void sigalarm(int i)
{
  // Update (will reach CursesLogger::update())
  dolfin_update();

  // Set new alarm
  alarm(1);
}

//-----------------------------------------------------------------------------
CursesLogger::CursesLogger() : GenericLogger()
{
  // Notify that we are still running the program
  finished = false;

  // Set state
  state = BUFFER;
  
  // Init curses
  win = initscr();

  // Get the number of rows and columns
  lines = LINES;
  cols  = COLS;

  // Do not display typed characters
  noecho();
  // Don't wait for newline, get characters one by one
  cbreak();
  // Don't wait if no character is typed
  nodelay(win, true);

  // Initialise colors
  if ( !has_colors() )
    dolfin_warning("Your terminal has no colors.");
  start_color();
  init_pair(1, COLOR_BLUE,  COLOR_WHITE);
  init_pair(2, COLOR_GREEN, COLOR_WHITE);
  init_pair(3, COLOR_RED,   COLOR_WHITE);
  init_pair(4, COLOR_WHITE, COLOR_BLACK);
  init_pair(5, COLOR_WHITE, COLOR_BLUE);
  init_pair(6, COLOR_BLACK, COLOR_WHITE);

  // Initialise the buffer
  buffer.init(lines, cols);
  offset = 1;

  // Allocate the gui message
  guiinfo = new char[cols];
  setInfo("DOLFIN started.");

  // Clear the buffer window
  attron(COLOR_PAIR(1));
  clearLines();

  // Set signals to catch alarms
  setSignals();
  
  // Draw window
  redraw();
}
//-----------------------------------------------------------------------------
CursesLogger::~CursesLogger()
{
  // Make sure that we are finished (so update() will ignore 'q')
  finished = true;
  
  // Add an extra line to the buffer
  buffer.add("");
  
  // Write a message
  setInfo("DOLFIN finished. Press q to quit.");
  redraw();

  // Wait for 'q'
  nodelay(win, false);
  while ( getch() != 'q' );
  
  // End curses
  endwin();

  // Clear gui message
  delete [] guiinfo;

  // Write a message in plain text
  std::cout << "DOLFIN finished at " << date() << "." << std::endl;
}
//-----------------------------------------------------------------------------
void CursesLogger::info(const char* msg)
{
  buffer.add(msg);
  redraw();
}
//-----------------------------------------------------------------------------
void CursesLogger::debug(const char* msg, const char* location)
{
  //std::cout << "DOLFIN debug [" << location << "]: " << msg << std::endl;
}
//-----------------------------------------------------------------------------
void CursesLogger::warning(const char* msg, const char* location)
{
  //std::cout << "DOLFIN warning [" << location << "]: " << msg << std::endl;
}
//-----------------------------------------------------------------------------
void CursesLogger::error(const char* msg, const char* location)
{
  //std::cout << "DOLFIN error [" << location << "]: " << msg << std::endl;
  exit(1);
}
//-----------------------------------------------------------------------------
void CursesLogger::progress(const char* title, const char* label, real p)
{
  /*
  int N = DOLFIN_TERM_WIDTH - 15;
  int n = (int) (p*((double) N));
  
  // Print the title
  printf("| %s", title);
  for (int i = 0; i < (N-length(title)-1); i++)
	 printf(" ");
  printf("|\n");
  
  // Print the progress bar
  printf("|");
  for (int i = 0; i < n; i++)
	 printf("=");
  if ( n > 0 && n < N ) {
	 printf("|");
	 n++;
  }
  for (int i = n; i < N; i++)
	 printf("-");
  printf("| %.1f\%\n", 100.0*p);
  */
}
//-----------------------------------------------------------------------------
void CursesLogger::update()
{
  if ( finished ) {
    redraw();
    return;
  }
  
  // Check keyboard for command
  char c = getch();

  switch ( c ) {
  case 'q':
    setInfo("Really quit? [y/n]");
    redraw();
    if ( getYesNo() ) {
      endwin();
      std::cout << "DOLFIN stopped at " << date() << "." << std::endl;
      kill(getpid(), SIGKILL);
    }
    else
      setInfo("Quit cancelled.");
    break;
  case 'p':
    setInfo("Paused. Press any key to continue.");
    redraw();
    getAnyKey();
    setInfo("Running.");
    break;
  case 'a':
    state = ABOUT;
    setInfo("Paused. Press any key to continue.");
    redraw();
    getAnyKey();
    setInfo("Running.");
    state = BUFFER;
    break;
  case 'h':
    state = HELP;
    setInfo("Paused. Press any key to continue.");
    redraw();
    getAnyKey();
    setInfo("Running.");
    state = BUFFER;
    break;
  case ERR:
    ;
    break;
  default:
    setInfo("Unknown command.");
  }

  redraw();

}
//-----------------------------------------------------------------------------
void CursesLogger::setSignals()
{
  struct sigaction act;
  
  // Set the signal handler for alarm
  act.sa_handler = sigalarm;
  sigaction(SIGALRM, &act, 0);

  // Set alarm
  alarm(1);
}
//-----------------------------------------------------------------------------
void CursesLogger::setInfo(const char *msg)
{
  int i = 0;
  for (i = 0; msg[i] && i < cols; i++) {
    guiinfo[i] = msg[i];
  }
  guiinfo[i] = '\0';
}
//-----------------------------------------------------------------------------
void CursesLogger::clearLines()
{
  for (int i = offset; i < (lines - 2); i++)
    clearLine(i,0);
}
//-----------------------------------------------------------------------------
void CursesLogger::clearLine(int line, int col)
{
  move(line, col);
  for (int i = col; i < cols; i++)
    printw(" ");
  move(line,col);
}
//-----------------------------------------------------------------------------
bool CursesLogger::getYesNo()
{
  // Wait for input
  nodelay(win, false);

  char c;
  while ( true ) {
    c = getch();
    switch ( c ) {
    case 'y':
      // Reset input state to normal
      nodelay(win, true);
      return true;
      break;
    case 'n':
      // Reset input state to normal
      nodelay(win, true);
      return false;
      break;
    default:
      setInfo("Please press 'y' or 'n'.");
      redraw();
    }
  }
}
//-----------------------------------------------------------------------------
void CursesLogger::getAnyKey()
{
  // Wait for input
  nodelay(win, false);

  // Get a character
  char c = getch();

  // Reset input state to normal
  nodelay(win, true);
}
//-----------------------------------------------------------------------------
void CursesLogger::drawTitle()
{
  attron(A_BOLD);
  attron(COLOR_PAIR(5));
  clearLine(0,0);
  printw("DOLFIN version %s", DOLFIN_VERSION);
}
//-----------------------------------------------------------------------------
void CursesLogger::drawBuffer()
{
  attron(A_NORMAL);

  int line = offset + buffer.size() - 1;
  if ( line > (lines - 3) )
    line = lines - 3;
  for (int i = buffer.size() - 1; i >= 0; i--) {

    // Print line from the buffer
    attron(COLOR_PAIR(1));
    clearLine(line, 0);
    printw(buffer.get(i));

    // Step to previous line
    line--;

    if ( line < offset )
      break;

  }
}
//-----------------------------------------------------------------------------
void CursesLogger::drawAbout()
{
  attron(A_NORMAL);
  attron(COLOR_PAIR(6));
  clearLines();

  move(offset + 1, 2);  printw(" ___   ___  _    ___ ___ _  _ ");
  move(offset + 2, 2);  printw("|   \\ / _ \\| |  | __|_ _| \\| |");
  move(offset + 3, 2);  printw("| () | (_) | |__| _| | ||  ` |");
  move(offset + 4, 2);  printw("|___/ \\___/|____|_| |___|_|\\_|");

  move(offset + 6, 2);  printw("DOLFIN is written and maintained by");
  move(offset + 8, 2);  printw("Johan Hoffman (hoffman@cims.nyu.edu)");
  move(offset + 9, 2);  printw("Anders Logg   (logg@math.chalmers.se)");
  
  move(offset + 11, 2); printw("For a complete list of author/contributors, see");
  move(offset + 12, 2); printw("the file AUTHORS in the source distribution.");

  move(offset + 14, 2); printw("For more information about DOLFIN, please visit");
  move(offset + 15, 2); printw("the web page at");
  move(offset + 17, 2); printw("http://www.phi.chalmers.se/dolfin/");
}
//-----------------------------------------------------------------------------
void CursesLogger::drawHelp()
{
  attron(A_NORMAL);
  attron(COLOR_PAIR(6));
  clearLines();

  move(offset + 1, 2);
  printw("No help available yet.");
}
//-----------------------------------------------------------------------------
void CursesLogger::drawInfo()
{
  attron(A_NORMAL);
  attron(COLOR_PAIR(5));
  clearLine(lines - 2, 0);
  printw(guiinfo);
}
//-----------------------------------------------------------------------------
void CursesLogger::drawCommands()
{
  attron(A_NORMAL);
  attron(COLOR_PAIR(4));

  clearLine(lines - 1, 0);
  
  printw("[q] quit [p] pause [a] about [h] help");
}
//-----------------------------------------------------------------------------
void CursesLogger::redraw()
{
  // Draw title
  drawTitle();

  // Draw contents
  switch ( state ) {
  case BUFFER:
    drawBuffer();
    break;
  case ABOUT:
    drawAbout();
    break;
  default:
    drawHelp();
    break;
  }

  // Draw info field
  drawInfo();

  // Draw command list
  drawCommands();

  // Move cursor to lower right corner
  move(lines - 1, cols - 1);

  // Call curses to refresh the window
  refresh();
}
//-----------------------------------------------------------------------------
