#include <dolfin.h>
#include <dolfin/Terminal.h>

Display *display = 0;

static bool initialized = false;

//-----------------------------------------------------------------------------
void dolfin_init(int argc, char **argv)
{
  // Check if we have already initialized
  if ( initialized ){
	 display->Warning("Already initialized.");
	 return;
  }
  
  // Do this only once
  initialized = true;

  // Initialize display
  display = new Terminal(0);
  
  // Write a comment
  display->Message(0,"Initializing DOLFIN.");
}
//-----------------------------------------------------------------------------
void dolfin_end()
{
  // Check that we have called the init function before this
  if ( !initialized )
	 display->Error("You must call dolfin_init() before dolfin_end().");
  
  // Write a comment
  display->Message(0,"Closing DOLFIN.");

  // Delete display
  delete display;
}
//-----------------------------------------------------------------------------
