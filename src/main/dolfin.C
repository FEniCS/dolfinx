#include "dolfin.h"
#include "Display.hh"
#include "Terminal.hh"
//#include "Curses.hh"
#include "Grid.hh"
#include "dolfin_modules.h"

// Temporary for testing
#include "Node.hh"
#include "List.hh"

// Database for all parameters
Settings *settings = NULL;

// The grid
Grid *grid;

// The problem
Problem *problem;

// Display is by default initialized as Terminal, since we need to be able
// to print messages before the initialization of the parameters, which
// in turn determine the type of Display.
Display *display = new Terminal(DOLFIN_INITIAL_DEBUG_LEVEL);

static bool initialized = false;

//-----------------------------------------------------------------------------
void dolfin_init(int argc, char **argv)
{
  // Check if we have already initialized
  if ( initialized ){
	 display->Warning("Already initialized");
	 return;
  }
  
  // Check that we have specified the problem
  if ( !settings )
 	 display->Error("Problem must be specified before initializing.");
  
  // Do this only once
  initialized = true;
  
  // Write a comment
  display->Message(0,"Initializing DOLFIN.");

  // Initialize all settings
  settings->Initialize();
  
  // Initialize the display
  int debug_level;
  dolfin_get_parameter("debug level",&debug_level);
  char display_type[DOLFIN_LINELENGTH];
  dolfin_get_parameter("display",display_type);
  if ( strcasecmp(display_type,"terminal") == 0 ){
	 delete display;
	 display = new Terminal(debug_level);
  }
  else if ( strcasecmp(display_type,"curses") == 0 ){
	 delete display;

	 display->Warning("Curses display not available. Using \"terminal\".");
         display = new Terminal(debug_level);

	//	 display = new Curses(debug_level);
  }
  else{
	 delete display;
	 display = new Terminal(debug_level);
	 display->Warning("Unknown display type \"%s\". Using \"terminal\".",display_type);
  }

  // Initialise the grid
  grid = new Grid;
  char gridfile[DOLFIN_LINELENGTH];
  dolfin_get_parameter("grid file",gridfile);
  grid->Read(gridfile);
  grid->Display();
  grid->Init();

  // Initialize problem
  char keyword[DOLFIN_PARAMSIZE];
  settings->Get("problem",keyword);
  problem = dolfin_module_problem(keyword,grid);
  
}
//-----------------------------------------------------------------------------
void dolfin_end()
{
  // Check that we have called the init function before this
  if ( !initialized )
	 display->Error("You must call dolfin_init() before dolfin_end().");

  // Delete the grid
  delete grid;

  // Delete the settings
  delete settings;

  // Delete the problem
  delete problem;
  
  // Write a comment
  display->Message(0,"Closing DOLFIN.");

  // Delete display
  delete display;
}
//-----------------------------------------------------------------------------
void dolfin_solve()
{
  // Check that we have called the init function before this
  if ( !initialized )
	 display->Error("You must call dolfin_init() before dolfin_solve()."); 
  
  int solve_primal, solve_dual;

  dolfin_get_parameter("solve primal",&solve_primal);
  dolfin_get_parameter("solve dual",&solve_dual);
 
  display->Message(0,"Using module: %s",problem->Description());

  problem->Solve();

  // Solve dual problem
  //if ( solve_dual )
  //dual->Solve();
} 
//-----------------------------------------------------------------------------
void dolfin_set_problem(const char *keyword)
{
  // Check that we have *not* specified the problem type
  if ( settings )
	 display->Error("Problem is already specified.");

  // Initialise settings
  settings = dolfin_module_settings(keyword);

  // Save problem keyword
  settings->Set("problem",keyword);
}
//-----------------------------------------------------------------------------
void dolfin_set_boundary_conditions(dolfin_bc (*bc)(real x, real y, real z,
														  int node, int component))
{
  // Check that we have specified the problem type
  if ( !settings )
    display->Error("You must specify problem before boundary conditions.");

  settings->bc_function = bc;
}
//-----------------------------------------------------------------------------
void dolfin_set_parameter(const char *identifier, ...)
{
  // Check that we have specified the problem type
  if ( !settings )
	 display->Error("Problem must be specified before setting parameters.");
  
  // Check that we have *not* called the init function before this
  if ( initialized ){
	 display->Warning("Problem already initialized. Unable to change settings.");
	 return;
  }
  
  va_list aptr;
  va_start(aptr,identifier);
  
  settings->SetByArgumentList(identifier,aptr);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void dolfin_get_parameter(const char *identifier, ...)
{
  // Check that we have specified the problem type
  if ( !settings )
	 display->Error("Problem must be specified before accessing parameters.");

  // Check that we have called the init function before this
  if ( !initialized )
	 display->Error("Unable to access parameters before initialization.");
  
  va_list aptr;
  va_start(aptr,identifier);
  
  settings->GetByArgumentList(identifier,aptr);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void dolfin_save_parameters()
{
  // Check that we have called the init function before this
  if ( !initialized ){
	 display->Error("Unable to save parameters before initialization.");
	 return;
  }

  settings->Save();
}
//-----------------------------------------------------------------------------
void dolfin_save_parameters(const char *filename)
{
  // Check that we have called the init function before this
  if ( !initialized ){
	 display->Error("Unable to save parameters before initialization.");
	 return;
  }
  
  settings->Save(filename);
}
//-----------------------------------------------------------------------------
void dolfin_load_parameters()
{
  // Check that we have *not* called the init function before this
  if ( initialized ){
	 display->Warning("Problem already initialized. Unable to load parameters.");
	 return;
  }
    
  settings->Load();
}
//-----------------------------------------------------------------------------
void dolfin_load_parameters(const char *filename)
{
  // Check that we have *not* called the init function before this
  if ( initialized ){
	 display->Warning("Problem already initialized. Unable to load parameters.");
	 return;
  }
  
  settings->Load(filename);
}
//-----------------------------------------------------------------------------
void dolfin_set_function (const char *identifier,
								  real (*f)(real x, real y, real z, real t))
{
  // Check that we have specified the problem type
  if ( !settings )
    display->Error("You must specify problem before setting a function.");

  settings->SetFunction(identifier,f);
}
//-----------------------------------------------------------------------------
void dolfin_test()
{
  cout << "Testing DOLFIN:" << endl;

  // Startup
  settings = new Settings();
  settings->Initialize();

  // Testing of DOLFIN internals
  
  List<Node> list;
  Node n;
  
  for (int i=0;i<77;i++){
	 n.SetNodeNo(i*3);
  	 list.add(n);
  }

  cout << list << endl;
  
  for (List<Node>::Iterator it = list.begin(); it != list.end(); ++it){
	 cout << it << endl;
	 cout << "*it = " << *it << endl;
	 cout << "it->GetNodeNo() = " << it->GetNodeNo() << endl;
	 cout << "it->pointer() = " << it.pointer() << endl;
  }
  
  //cout << list << endl;
  
  // Cleanup
  delete settings;
  delete display;
}
//-----------------------------------------------------------------------------
void dolfin_test_memory()
{
  // Use this function for testing memory. If this fails (i.e. gives
  // segmentation fault) then something is probably wrong before the call
  // to this function...

  display->Message(0,"DOLFIN memory test...");

  char **test;
  int n = 100;
  
  test = new (char *)[n];
  for (int i=0;i<n;i++){
	 test[i] = new char[DOLFIN_LINELENGTH];
	 sprintf(test[i],"data component %d",i);
  }
  
  for (int i=0;i<n;i++)
	 delete test[i];
  delete test;

  display->Message(0,"Memory test ok.");
}
//-----------------------------------------------------------------------------
