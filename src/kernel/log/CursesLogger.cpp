// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef NO_CURSES

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

// A note on the update process: Every second, an alarm is triggered and
// the function update() is called. We then remain in the update loop
// until it is time to start running again.
//
// A warning: don't use sleep(). Not a good idea to mix sleep() and alarm().

// Signal handler, alarm every second as long as the computation is running
void sigalarm(int i)
{
  // Update (will reach CursesLogger::update())
  dolfin_update();
  
  // Set new alarm if the computation is still running
  if ( !dolfin_finished() )
    alarm(1);
}

// Signal handler, ctrl-c
void sigctrlc(int i)
{
  // Quit (will reach CursesLogger::quit())
  dolfin_quit();
}

//-----------------------------------------------------------------------------
CursesLogger::CursesLogger() : GenericLogger()
{
  // Set state
  state = RUNNING;

  // Computation still running
  running = true;

  // Not updating
  updating = false;

  // Not waiting for input
  waiting = false;
  
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

  // Initialise progress bars
  pbars = new Progress * [DOLFIN_PROGRESS_BARS];
  for (int i = 0; i < DOLFIN_PROGRESS_BARS; i++)
    pbars[i] = 0;

  // Initialise time to display progress bars (will remain for a few seconds after removed)
  ptimes = new int [DOLFIN_PROGRESS_BARS];
  for (int i = 0; i < DOLFIN_PROGRESS_BARS; i++)
    ptimes[i] = -1;

  // Initialise the buffer
  buffer.init(lines, cols);
  offset = 1;

  // Allocate the gui message
  guiinfo = new char[cols];
  setInfo("DOLFIN started.");

  // Allocate temporary storage
  tmp = new char[DOLFIN_WORDLENGTH];
  tmp[0] = '\0';

  // Clear the buffer window
  attron(COLOR_PAIR(1));
  clearLines();

  // Set signals to catch alarms
  setSignals();
  
  // Initialise colors
  initColors();

  // Draw window
  redraw();
}
//-----------------------------------------------------------------------------
CursesLogger::~CursesLogger()
{
  // Make sure that we are finished 
  state = FINISHED;
  running = false;
  
  // Add an extra line to the buffer
  buffer.add("");
  
  // Write a message
  setInfo("DOLFIN finished. Press q to quit.");
  redraw();

  // Wait for 'q'
  updateInternal();

  // End curses
  endwin();

  // Clear gui message
  if ( guiinfo )
    delete [] guiinfo;
  guiinfo = 0;

  // Clear temporary storage
  if ( tmp )
    delete [] tmp;
  tmp = 0;

  // Clear progress bars
  if ( pbars )
    delete [] pbars;
  pbars = 0;

  // Clear remaing times for progress bars
  if ( ptimes )
    delete [] ptimes;
  ptimes = 0;

  // Write a message in plain text
  std::cout << "DOLFIN finished at " << date() << "." << std::endl;
}
//-----------------------------------------------------------------------------
void CursesLogger::info(const char* msg)
{
  buffer.add(msg, Buffer::info, level);
  redraw();
}
//-----------------------------------------------------------------------------
void CursesLogger::debug(const char* msg, const char* location)
{
  snprintf(tmp, DOLFIN_WORDLENGTH, "Debug at %s: %s", location, msg);
  buffer.add(tmp, Buffer::debug, level);
  redraw();
}
//-----------------------------------------------------------------------------
void CursesLogger::warning(const char* msg, const char* location)
{
  snprintf(tmp, DOLFIN_WORDLENGTH, "Warning at %s: %s", location, msg);
  buffer.add(tmp, Buffer::warning, level);
  redraw();
}
//-----------------------------------------------------------------------------
void CursesLogger::error(const char* msg, const char* location)
{
  snprintf(tmp, DOLFIN_WORDLENGTH, "Error at %s: %s", location, msg);
  buffer.add(tmp, Buffer::error, level);
  setInfo("Press any key to quit.");
  state = ERROR;
  updateInternal();
}
//-----------------------------------------------------------------------------
void CursesLogger::dassert(const char* msg, const char* location)
{
  snprintf(tmp, DOLFIN_WORDLENGTH, "Assertion %s failed at %s", msg, location);
  buffer.add(tmp, Buffer::error, level);
  setInfo("Press any key to quit.");
  state = ERROR;
  updateInternal();
}
//-----------------------------------------------------------------------------
void CursesLogger::progress(const char* title, const char* label, real p)
{
  updateInternal();
}
//----------------------------------------------------------------------------- 
void CursesLogger::update()
{
  // Updates that should be done periodically
  progress_flush();
  
  // Other updates
  updateInternal();
}
//----------------------------------------------------------------------------- 
void CursesLogger::quit()
{
  endwin();
  std::cout << "DOLFIN stopped at " << date() << "." << std::endl;
  kill(getpid(), SIGKILL);
}
//-----------------------------------------------------------------------------
bool CursesLogger::finished()
{
  return state == FINISHED;
}
//-----------------------------------------------------------------------------
void CursesLogger::progress_add(Progress* p)
{
  for (int i = 0; i < DOLFIN_PROGRESS_BARS; i++)
    if ( ptimes[i] != -1 ) { // Use an old one that we should remove
      pbars[i] = p;
      ptimes[i] = -1;
      redraw();
      return;
    }
    else if ( pbars[i] == 0 ) { // Create a new progress bar
      pbars[i] = p;
      ptimes[i] = -1;
      offset += 2;
      redraw();
      return;
    }
  
  dolfin_warning("Too many progress bars.");
  redraw();
}
//-----------------------------------------------------------------------------
void CursesLogger::progress_remove(Progress *p)
{ 
  // Find the progress bar in the list and schedule to be removed after
  // DOLFIN_PROGRESS_WAIT seconds

  for (int i = 0; i < DOLFIN_PROGRESS_BARS; i++)
    if ( pbars[i] == p ) {
      pbars[i] = 0;
      ptimes[i] = DOLFIN_PROGRESS_WAIT;
      return;
    }
  
  dolfin_warning("Removing unknown  progress.bar.");
  redraw();
}
//-----------------------------------------------------------------------------
void CursesLogger::progress_flush()
{
  // Remove any progress bars scheduled to be removed
  
  for (int i = 0; i < DOLFIN_PROGRESS_BARS;) {

    // Check if we should skip this progress bar
    if ( ptimes[i] == -1 ) {
      i++;
      continue;
    }
    
    // Decrease the number of remaining seconds
    ptimes[i] -= 1;
    
    // Check if the progress bar should be removed
    if ( ptimes[i] == 0 ) {
      for (int j = i + 1; j < DOLFIN_PROGRESS_BARS; j++) {
	pbars[j-1] = pbars[j];
	ptimes[j-1] = ptimes[j];
      }
      pbars[DOLFIN_PROGRESS_BARS - 1] = 0;
      ptimes[DOLFIN_PROGRESS_BARS - 1] = -1;
      offset -= 2;
      continue;
    }

    // Jump to next progress bar
    i++;
    
  }

}
//----------------------------------------------------------------------------- 
void CursesLogger::initColors()
{
  if ( !has_colors() ) {
    buffer.add("Your terminal has no colors.", Buffer::warning, level);
    return;
  }

  start_color();

  //if ( !can_change_color() )
  //  buffer.add("Unable to change colors. Using default colors.", Buffer::warning);
  //else {
  //  init_color(COLOR_BLUE,  0, 0, 1000);
  //  init_color(COLOR_GREEN, 0, 1000, 0);
  //  init_color(COLOR_RED,   1000, 0, 0);
  //  init_color(COLOR_WHITE, 1000, 1000, 1000);
  //  init_color(COLOR_BLACK, 0, 0, 0);
  //}

  init_pair(1, COLOR_BLUE,  COLOR_WHITE);
  init_pair(2, COLOR_GREEN, COLOR_WHITE);
  init_pair(3, COLOR_RED,   COLOR_WHITE);
  init_pair(4, COLOR_WHITE, COLOR_BLACK);
  init_pair(5, COLOR_WHITE, COLOR_BLUE);
  init_pair(6, COLOR_BLACK, COLOR_WHITE);
}
//----------------------------------------------------------------------------- 
void CursesLogger::updateInternal()
{
  // Ignore call if we're waiting for input
  if ( waiting )
    return;

  // Ignore call if we're already updating
  if ( updating )
    return;

  // Block other calls to this function
  updating = true;

  while ( true ) {
    
    // Check keyboard for command (non-locking input selected in constructor)
    char c = getch();
    
    if ( c != ERR ) {
      
      switch ( state ) {
      case RUNNING:
	updateRunning(c);
	break;
      case PAUSED:
	updatePaused(c);
	break;
      case ERROR:
	updateError(c);
	break;
      case ABOUT:
	updateAbout(c);
	break;
      case HELP:
	updateHelp(c);
	break;
      default: // FINISHED
	updateFinished(c);
      }

    }
    
    redraw();
    
    // Check if we should exit the loop
    if ( state == RUNNING || state == QUIT )
      break;

    // Sleep for a while
    delay(0.01);

  }

  // Remove block to this function
  updating = false;
}
//-----------------------------------------------------------------------------
void CursesLogger::updateRunning(char c)
{
  switch ( c ) {
  case 'q':
    if ( running )
      killProgram();
    else {
      setInfo("DOLFIN finished.");
      state = QUIT;
    }
    break;
  case 'p':
    if ( running ) {
      setInfo("Paused. Press space or 'p' to continue.");
      state = PAUSED;
    }
    else 
      setInfo("Program has finished.");
    break;
  case 'a':
    if ( running ) {
      setInfo("Press space to continue.");
      state = ABOUT;
    }
    else {
      setInfo("Press space to return.");
      state = ABOUT;
    }
    break;
  case 'h':
    if ( running ) {
      setInfo("Press space to continue.");
      state = HELP;
    }
    else {
      setInfo("Press space to return.");
      state = HELP;
    }
    break;
  default:
    setInfo("Unknown command.");
  }
}
//-----------------------------------------------------------------------------
void CursesLogger::updatePaused(char c)
{
  switch ( c ) {
  case 'q':
    killProgram();
    break;
  case 'p':
    if ( running ) {
      setInfo("Running.");
      state = RUNNING;
    }
    else
      setInfo("Program paused but has finished. Weird.");
    break;
  case 'a':
    setInfo("Press space to continue.");
    state = ABOUT;
    break;
  case 'h':
    setInfo("Press space to continue.");
    state = HELP;
    break;
  case ' ':
    setInfo("Running.");
    state = RUNNING;
    break;
  default:
    setInfo("Unknown command.");
  }
}
//-----------------------------------------------------------------------------
void CursesLogger::updateError(char c)
{
  endwin();
  std::cout << "DOLFIN stopped at " << date() << "." << std::endl;
  kill(getpid(), SIGKILL);
}
//-----------------------------------------------------------------------------
void CursesLogger::updateAbout(char c)
{
  switch ( c ) {
  case 'q':
    if ( running )
      killProgram();
    else {
      setInfo("DOLFIN finished.");
      state = QUIT;
      return;
    }
    break;
  case 'p':
    if ( running ) 
      setInfo("Already paused.");
    else
      setInfo("Program has finished.");
    break;
  case 'a':
    // Ignore
    break;
  case 'h':
    if ( running ) {
      setInfo("Press space to continue.");
      state = HELP;
    }
    else {
      setInfo("Press space to return.");
      state = HELP;
    }
    break;
  case ' ':
    if ( running ) {
      setInfo("Running.");
      state = RUNNING;
    }
    else {
      setInfo("DOLFIN finished.");
      state = FINISHED;
    }
    break;
  default:
    setInfo("Unknown command.");
  }
}
//-----------------------------------------------------------------------------
void CursesLogger::updateHelp(char c)
{
  switch ( c ) {
  case 'q':
    if ( running )
      killProgram();
    else {
      setInfo("DOLFIN finished.");
      state = QUIT;
    }
    break;
  case 'p':
    if ( running ) 
      setInfo("Already paused.");
    else
      setInfo("Program has finished.");
    break;
  case 'a':
    if ( running ) {
      setInfo("Press space to continue.");
      state = ABOUT;
    }
    else {
      setInfo("Press space to return.");
      state = ABOUT;
    }
    break;
  case 'h':
    // Ignore
    break;
  case ' ':
    if ( running ) {
      setInfo("Running.");
      state = RUNNING;
    }
    else {
      setInfo("DOLFIN finished.");
      state = FINISHED;
    }
    break;
  default:
    setInfo("Unknown command.");
  }
}
//-----------------------------------------------------------------------------
void CursesLogger::updateFinished(char c)
{
  switch ( c ) {
  case 'q':
    state = QUIT;
    break;
  case 'p':
    setInfo("Program has finished.");
    break;
  case 'a':
    setInfo("Press space to return.");
    state = ABOUT;
    break;
  case 'h':
    setInfo("Press space to return.");
    state = HELP;
    break;
  default:
    setInfo("Unknown command.");
  }
}
//-----------------------------------------------------------------------------
void CursesLogger::killProgram()
{
  setInfo("Stop running program? [y/n]");
  redraw();
  
  if ( getYesNo() ) {
    endwin();
    std::cout << "DOLFIN stopped at " << date() << "." << std::endl;
    kill(getpid(), SIGKILL);
  }
  else
    setInfo("Quit cancelled.");
}
//-----------------------------------------------------------------------------
void CursesLogger::setSignals()
{
  struct sigaction act;
  
  // Set the signal handler for alarm
  sigaction(SIGALRM, 0, &act);
  act.sa_handler = sigalarm;
  sigaction(SIGALRM, &act, 0);

  // Set signal handler for ctrl-c
  sigaction(SIGALRM, 0, &act);
  act.sa_handler = sigctrlc;
  sigaction(SIGINT, &act, 0);

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
  for (int i = 1; i < (lines - 2); i++)
    clearLine(i,0);
}
//-----------------------------------------------------------------------------
void CursesLogger::clearBuffer()
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
  // Notify that we are waiting for input
  waiting = true;

  // Wait for input
  nodelay(win, false);

  char c;
  while ( true ) {
    c = getch();
    switch ( c ) {
    case 'y':
      // Reset input state to normal
      nodelay(win, true);
      waiting = false;
      return true;
      break;
    case 'n':
      // Reset input state to normal
      nodelay(win, true);
      waiting = false;
      return false;
      break;
    default:
      // Ignore (seems like getch() stops waiting for input at alarm)
      ;
    }
  }

  // Reset input state to normal
  nodelay(win, true);
  waiting = false;
}
//-----------------------------------------------------------------------------
void CursesLogger::getAnyKey()
{
  // Notify that we are waiting for input
  waiting = true;

  // Wait for input
  nodelay(win, false);

  // Get a character
  getch();

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
void CursesLogger::drawProgress()
{
  attroff(A_BOLD);
  attron(COLOR_PAIR(4));
  int line = 0;
  int n = 0;
  real p = 0.0;

  for (int i = 0; i < DOLFIN_PROGRESS_BARS; i++) {    

    // Skip if not in use
    if ( pbars[i] == 0 && ptimes[i] == -1 )
      continue;

    // Get value if not removed
    if ( pbars[i] != 0 )
      p = pbars[i]->value();
    else
      p = 1.0;
   
    line = 1 + 2*i;
    
    // Draw title
    if ( pbars[i] != 0 ) {
      clearLine(line, 0);
      printw("%s", pbars[i]->title());
    }

    // Draw progress bar
    n = (int) (p * ((real) cols));
    clearLine(line + 1, 0);
    for (int j = 0; j < n; j++)
      printw("=");
    if ( n < cols )
      printw("|");
    for (int j = n + 1; j < cols; j++)
      printw("-");
    
    // Draw progress value
    move(line, cols - 5);
    printw("%2.1f", 100.0 * p);
    move(line, cols - 1);
    printw("%%");

  }

}
//-----------------------------------------------------------------------------
void CursesLogger::drawBuffer()
{
  attroff(A_BOLD);
  attron(COLOR_PAIR(6));
  clearBuffer();
  
  int line = offset + buffer.size() - 1;
  if ( line > (lines - 3) )
    line = lines - 3;

  for (int i = buffer.size() - 1; i >= 0; i--) {
    
    // Set color
    switch ( buffer.type(i) ) {
    case Buffer::info:
      attron(COLOR_PAIR(6));
      break;
    case Buffer::debug:
      attron(COLOR_PAIR(1));
      break;
    case Buffer::warning:
      attron(COLOR_PAIR(2));
      break;
    default: // ERROR
      attron(COLOR_PAIR(3));
    }

    // Print line from the buffer
    clearLine(line, 0);
    indent(buffer.level(i));
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
  attroff(A_BOLD);
  attron(COLOR_PAIR(6));
  clearLines();
  
  int line = 2;
  
  move(line +  0, 2);  printw("DOLFIN is written and maintained by");
  move(line +  1, 2);  printw("Johan Hoffman (hoffman@cims.nyu.edu)");
  move(line +  2, 2);  printw("Anders Logg   (logg@math.chalmers.se)");
  
  move(line +  4, 2); printw("For a complete list of authors/contributors, see");
  move(line +  5, 2); printw("the file AUTHORS in the source distribution.");

  move(line +  7, 2); printw("For more information about DOLFIN, please visit");
  move(line +  8, 2); printw("the web page at http://www.phi.chalmers.se/dolfin/");
}
//-----------------------------------------------------------------------------
void CursesLogger::drawHelp()
{
  attroff(A_BOLD);
  attron(COLOR_PAIR(6));
  clearLines();

  int line = 2;

  move(line + 0,  2); printw("Shortcut keys");
  move(line + 1,  2); printw("-------------");

  move(line + 3,  2); printw("q - press to quit / kill process");
  move(line + 4,  2); printw("p - press to toggle pause");
  move(line + 5,  2); printw("a - press to display some info about DOLFIN");
  move(line + 6,  2); printw("h - press to display this help");

  move(line + 8,  2); printw("In preparation");
  move(line + 9,  2); printw("--------------");
  move(line + 11, 2); printw("- Scrolling using arrow keys");
  move(line + 12, 2); printw("- Stepping program line by line");
  move(line + 13, 2); printw("- Web browser and coffee-making ;)");
}
//-----------------------------------------------------------------------------
void CursesLogger::drawInfo()
{
  attroff(A_BOLD);
  attron(COLOR_PAIR(5));
  clearLine(lines - 2, 0);
  printw(guiinfo);
}
//-----------------------------------------------------------------------------
void CursesLogger::drawCommands()
{
  attroff(A_BOLD);
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
  case RUNNING:
    drawProgress();
    drawBuffer();
    break;
  case ABOUT:
    drawAbout();
    break;
  case HELP:
    drawHelp();
    break;
  default: // FINISHED
    drawBuffer();
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
void CursesLogger::indent(int _level)
{
  // Indent output to indicate the level. Note that we need to use the level
  // stored for each row in the buffer and not the current level.

  for (int i = 0; i < _level; i++)
    printw("  ");
}
//-----------------------------------------------------------------------------

#endif
