// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "Curses.hh"
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <curses.h>

//-----------------------------------------------------------------------------
Curses::Curses(int debug_level) : Display(debug_level)
{
  // Initialize curses
  initscr();
}
//-----------------------------------------------------------------------------
Curses::~Curses()
{
  // End curses
  endwin();
}
//-----------------------------------------------------------------------------
void Curses::Status(int level, const char *format, ...)
{
  switch (level){
  case 0:
	 printf(":::: ");
	 break;
  case 1:
	 printf(":::: ::: ");
	 break;
  case 2:
	 printf(":::: ::: :: ");
	 break;
  default:
	 printf(":::: ::: :: : ");
  }

  va_list aptr;
  va_start(aptr,format);

  vprintf(format,aptr);
  printf("\n");
  
  va_end(aptr);  
}
//-----------------------------------------------------------------------------
void Curses::Message(int level, const char *format, ...)
{
  va_list aptr;
  va_start(aptr,format);

  if ( (level >= debug_level) || (debug_level < 0) )
	 vprintf(format,aptr);
  
  va_end(aptr);  
}
//-----------------------------------------------------------------------------
void Curses::Progress(int level, double progress, const char *format, ...)
{
  va_list aptr;
  va_start(aptr,format);

  char description[KW_LINELENGTH];
  vsprintf(description,format,aptr);
  
  va_end(aptr);  

  move(2*level,0);
  printw("|");
  move(2*level,2);
  printw(description);
  move(2*level,COLS-1);
  printw("|");
  
  if ( progress < 0.0 )
	 progress = 0.0;
  if ( progress > 1.0 )
	 progress = 1.0;
  
  ProgressBar(2*level+1,progress);
}
//-----------------------------------------------------------------------------
void Curses::Regress(int level, double progress, double maximum,
							const char *format, ...)
{
  va_list aptr;
  va_start(aptr,format);

  char description[KW_LINELENGTH];
  vsprintf(description,format,aptr);
  
  va_end(aptr);  

  move(2*level,0);
  printw("|");
  move(2*level,2);
  printw(description);
  move(2*level,COLS-1);
  printw("|");
  
  if ( progress < 0.0 )
	 progress = 0.0;
  if ( progress > 1.0 )
	 progress = 1.0;
  
  RegressBar(2*level+1,progress,maximum);
}
//-----------------------------------------------------------------------------
void Curses::Value(const char *name, Type type, ...)
{
  va_list aptr;
  va_start(aptr,type);

  double val_double;
  int    val_int;
  bool   val_bool;
  char  *val_string;

  switch (type){
  case type_double:
	 val_double = va_arg(aptr,double);
	 printf("%s = %f\n",name,val_double);
	 break;
  case type_int:
	 val_int = va_arg(aptr,int);
	 printf("%s = %d\n",name,val_int);
	 break;
  case type_bool:
	 val_bool = va_arg(aptr,bool);
	 printf("%s = %s\n",name,(val_bool ? "true" : "false"));
	 break;
  case type_string:
	 val_string = va_arg(aptr,char *);
	 printf("%s = \"%s\"\n",name,val_string);
	 break;
  default:
	 InternalError("Terminal:Value","Unknown type.");
  }
  
  va_end(aptr);  
}
//-----------------------------------------------------------------------------
void Curses::Warning(const char *format, ...)
{
  va_list aptr;
  va_start(aptr,format);
  va_end(aptr);  
  
  printf("*** DOLFIN warning: ");
  vprintf(format,aptr);
  printf("\n");
  printf("*** Trying to continue anyway. (This might not work.)\n");
}
//-----------------------------------------------------------------------------
void Curses::Error(const char *format, ...)
{
  va_list aptr;
  va_start(aptr,format);
  va_end(aptr);  
  
  printf("*** DOLFIN error: ");
  vprintf(format,aptr);
  printf("\n");
  printf("*** Exiting.\n");

  exit(1);
}
//-----------------------------------------------------------------------------
void Curses::InternalError(const char *function, const char *format, ...)
{
  va_list aptr;
  va_start(aptr,format);
  va_end(aptr);  
  
  printf("*** DOLFIN internal error in function %s: ",function);
  vprintf(format,aptr);
  printf("\n");
  printf("*** Exiting.\n");

  exit(2);
}
//-----------------------------------------------------------------------------
void Curses::ProgressBar(int row, double progress)
{
  // This function writes a progress bar.
  
  int i;
  
  // Fix progress
  if ( progress < 0.0 )
	 progress = 0.0;
  if ( progress > 1.0 )
	 progress = 1.0;
  
  move(row,0);
  printw("|");
  for (i=1;i<int(double(progress)*double(COLS))-1;i++){
 	 move(row,i);
 	 printw("=");
  }
  printw("|");
  for (i++;i<(COLS-1);i++){
 	 move(row,i);
 	 printw("-");
  }  
  move(row,COLS-1);
  printw("|");
  
}
//----------------------------------------------------------------------------
void Curses::RegressBar(int row, double progress, double maximum)
{
  // This function writes a regress bar, i.e. a progress bar starting from
  // the left. Could be used for the dual solution.
  
  int i;
	
  // Fix progress and maximum
  if ( progress < 0.0 )
	 progress = 0.0;
  if ( progress > 1.0 )
	 progress = 1.0;

  if ( maximum < 0.0 )
	 progress = 0.0;
  if ( progress > 1.0 )
	 progress = 1.0;
	 
  // Clear the bar
  for (int j=0;j<COLS;j++){
	 move(row,j);
	 printw(" ");
  }

  // Set the length of the bar
  int cols = int( double(COLS) * maximum );

  // Draw the bar
  move(row,cols-1);
  printw("|");
  for (i=1;i<int(progress*double(cols));i++){
 	 move(row,cols-i-1);
 	 printw("=");
  }
   move(row,cols-i);
   printw("|");
   for (i=cols-i-1;i>0;i--){
 	 move(row,i);
 	 printw("-");
   }  
   move(row,0);
   printw("|");

}
//----------------------------------------------------------------------------

// #ifdef HAVE_CURSES
//   // Initialize curses
//   initscr();
// #endif

//   // Write the logo
//   WriteLogo();
// }
// //----------------------------------------------------------------------------
// Output::~Output()
// {
//   // Don't write any output if we shouldn't do that
//   if ( iOutputWhere == ADDRESS_DEVNULL ) return;

// #ifdef HAVE_CURSES
//   // Finish curses
//   endwin();
// #endif

// }
// //----------------------------------------------------------------------------
// void Output::Message(char *cString, int iRow)
// {
//   // This function writes a message
  
//   // Don't write any output if we shouldn't do that
//   if ( iOutputWhere == ADDRESS_DEVNULL ) return;
  
// #ifdef HAVE_CURSES
//   // Print the "Message: " text
//   move(MESSAGE_ROW,2);
//   printw("Message:");
  
//   // Print message
//   move(MESSAGE_ROW+iRow,MESSAGE_COL);
//   printw("%s",cString);
  
//   // Force output
//   refresh();
// #endif 

// #ifndef HAVE_CURSES
//   cout << "  Message: " << cString << endl;
// #endif
  
// }
// //----------------------------------------------------------------------------
// void Output::MessageClear()
// {
//   // This function clears the messsage output.

//   // Don't write any output if we shouldn't do that
//   if ( iOutputWhere == ADDRESS_DEVNULL ) return;

// #ifdef HAVE_CURSES
//   for (int i=MESSAGE_ROW;i<(MESSAGE_ROW+MESSAGE_ROWS);i++)
//     for (int j=MESSAGE_COL;j<(COLS-1);j++){
//       move(i,j);
// 		printw(" ");
//     }
  
//   // Force output
//   refresh();
// #endif

// }
// //----------------------------------------------------------------------------
// void Output::Comment(char *cString)
// {
//   // This function writes a comment.

//   // Don't write any output if we shouldn't do that
//   if ( iOutputWhere == ADDRESS_DEVNULL ) return;
  
// #ifdef HAVE_CURSES
//   // Clear the previous comment
//   for (int i=COMMENT_COL;i<(COLS-1);i++){
// 	 move(COMMENT_ROW,i);
// 	 printw(" ");
//   }

//   // Print the "Comment:" text
//   move(COMMENT_ROW,2);
//   printw("Comment:");

//   // Print the comment
//   move(COMMENT_ROW,COMMENT_COL);
//   printw(cString); 

//   // Force output
//   refresh();
// #endif

// #ifndef HAVE_CURSES
//   cout << "  Comment: " << cString << endl;
// #endif  

// }
// //----------------------------------------------------------------------------
// void Output::Error(char *cString)
// {
//   // This function writes an error message.

// #ifdef HAVE_CURSES
//   // Finish curses
//   endwin();
// #endif
  
//   if ( iOutputWhere != ADDRESS_DEVNULL ){
//     cout << "Error: " << cString << endl;
//     cout << "Exiting" << endl;
//   }
  
//   exit(0);

// }
// //----------------------------------------------------------------------------
// void Output::Warning(char *cString)
// {
//   // This function writes a warning.

//   // Don't write any output if we shouldn't do that
//   if ( iOutputWhere == ADDRESS_DEVNULL ) return;

// #ifdef HAVE_CURSES  
//   // Clear the previous warning
//   for (int i=WARNING_COL;i<(COLS-1);i++){
//     move(WARNING_ROW,i);
//     printw(" ");
//   }
  
//   // Print the "Warning:" text
//   move(WARNING_ROW,2);
//   printw("Warning:");
  
//   // Print the warning in bold font
//   move(WARNING_ROW,WARNING_COL);
//   attrset(A_BOLD); 
//   printw(cString); 
  
//   // Change back to normal font
//   attrset(0); 
  
//   // Force output
//   refresh();
// #endif

// #ifndef HAVE_CURSES
//   cout << "  *** Error: " << cString << " ***" << endl;
// #endif
  
// }
// //----------------------------------------------------------------------------
// void Output::Progress(double dProgress_1,
// 							 double dProgress_2,
// 							 int    iiSizeOfSlab,
// 							 int    iiSizeCleared,
// 							 double ddNumberOfIterations,
// 							 double ddMaxConservation,
// 							 bool   bDualSolution)
// {
//   // This function writes nicely formatted info in the current terminal.
  
//   // Don't write any output if we shouldn't do that
//   if ( iOutputWhere == ADDRESS_DEVNULL ) return;

//   // Save the data
//   iSizeOfSlab         = iiSizeOfSlab;
//   iSizeCleared        = iiSizeCleared;
//   dNumberOfIterations = ddNumberOfIterations;
//   dMaxConservation    = ddMaxConservation;

// #ifdef HAVE_CURSES
//   // Write progress
//   move(0,1);
//   printw("Progress: %2.1f%%",100*dProgress_1);

//   // Write position of front
//   move(0,COLS-14);
//   printw("[Front %2.1f%%]",100*dProgress_2);

//   // Write the progress bar
//   if ( !bDualSolution )
// 	 ProgressBar(1,dProgress_1);
//   else
// 	 RegressBar(2,dProgress_1);
  
//   // Write the progress bar for the front
//   //ProgressBar(2,dProgress_2);

//   // Write some data
//   WriteData();
  
//   // Force output
//   refresh();
// #endif

// #ifndef HAVE_CURSES
//   cout << "  Progress: " << (dProgress_1*100.0) << " % ";
//   cout << "N = [ " << iSizeOfSystem << " " << iSizeOfSlab << " " << iSizeCleared << " ]" << endl;
// #endif

// }
// //----------------------------------------------------------------------------
// void Output::Status(int iiStatus) 
// {
//   // This function writes the current status.

//   // Don't write any output if we shouldn't do that
//   if ( iOutputWhere == ADDRESS_DEVNULL ) return;

// #ifdef HAVE_CURSES
//   int iRow     = 12;
//   int iCol     = 2;
//   int iColDone = 32;
  
//   switch (iiStatus){
//   case STATUS_CLEAR:
// 	 for (int i=0;i<9;i++){
// 		move(iRow+i,iCol);
// 		printw("                             ");
// 	 }
// 	 break;
//   case STATUS_FORWARD:
// 	 move(iRow+0,iCol);
// 	 printw("Computing solution...........");
// 	 break;
//   case STATUS_FORWARD_DONE:
// 	 move(iRow+0,iColDone);
// 	 printw("done");
// 	 break;
//   case STATUS_DUAL:
// 	 move(iRow+3,iCol);
// 	 printw("Solving dual problem.........");
// 	 break;
//   case STATUS_DUAL_DONE:
// 	 move(iRow+3,iColDone);
// 	 printw("done");
// 	 break;
//   case STATUS_ERROR_ESTIMATE:
// 	 move(iRow+7,iCol);
// 	 printw("Computing error estimate.....");
// 	 break;
//   case STATUS_ERROR_ESTIMATE_DONE:
// 	 move(iRow+7,iColDone);
// 	 printw("done");
// 	 break;
//   case STATUS_NEW_SOLUTION:
// 	 move(iRow+8,iCol);
// 	 printw("Preparing for new solution...");
// 	 break;
//   case STATUS_NEW_SOLUTION_DONE:
// 	 move(iRow+8,iColDone);
// 	 printw("done");
// 	 break;
//   default:
// 	 break;
//   }  

//   // Force output
//   refresh();
// #endif

// #ifndef HAVE_CURSES
//   cout << "================================================================" << endl;
//   switch (iiStatus){
//   case STATUS_CLEAR:
//     break;
//   case STATUS_FORWARD:
//     cout << "Status: Computing solution..........." << endl;
//     break;
//   case STATUS_FORWARD_DONE:
//     cout << "Status: done" << endl;
//     break;
//   case STATUS_DUAL:
//     cout << "Status: Solving dual problem........." << endl;
//     break;
//   case STATUS_DUAL_DONE:
//     cout << "Status: done" << endl;
//     break;
//   case STATUS_ERROR_ESTIMATE:
//     cout << "Status: Computing error estimate....." << endl;
//     break;
//   case STATUS_ERROR_ESTIMATE_DONE:
//     cout << "Status: done" << endl;
//     break;
//   case STATUS_NEW_SOLUTION:
//     cout << "Status: Preparing for new solution..." << endl;
//     break;
//   case STATUS_NEW_SOLUTION_DONE:
//     cout << "Status: done" << endl;
//     break;
//   default:
//     break;
//   }  
//   cout << "================================================================" << endl;
// #endif
  
//   // Save the status
//   iStatus = iiStatus;
  
// }
// //----------------------------------------------------------------------------
// void Output::SubStatus(int iiSubStatus)
// {
//   // This function writes the current substatus.

//   // Don't write any output if we shouldn't do that
//   if ( iOutputWhere == ADDRESS_DEVNULL ) return;
 
// #ifdef HAVE_CURSES 
//   int iRow     = 12;
//   int iCol     = 4;
//   int iColDone = 32;
  
//   switch (iiSubStatus){
//   case SUBSTATUS_INITTIMESTEPS:
// 	 if ( iStatus == STATUS_FORWARD )
// 		move(iRow+1,iCol);
// 	 else
// 	 	move(iRow+5,iCol);
// 	 printw("Initializing timesteps.....");
// 	 break;
//   case SUBSTATUS_INITTIMESTEPS_DONE:
// 	 if ( iStatus == STATUS_FORWARD )
// 		move(iRow+1,iColDone);
// 	 else
// 	 	move(iRow+5,iColDone);
// 	 printw("done");
// 	 break;
//   case SUBSTATUS_STEPPING:
// 	 if ( iStatus == STATUS_FORWARD )
// 		move(iRow+2,iCol);
// 	 else
// 		move(iRow+6,iCol);
// 	 printw("Stepping...................");
// 	 break;
//   case SUBSTATUS_STEPPING_DONE:
// 	 if ( iStatus == STATUS_FORWARD )
// 		move(iRow+2,iColDone);
// 	 else
// 		move(iRow+6,iColDone);
// 	 printw("done");
// 	 // Fix progress bars
// 	 if ( iStatus == STATUS_FORWARD )
// 		ProgressBar(1,1.0);
// 	 else
// 		RegressBar(2,1.0);
// 	 //ProgressBar(2,1.0);	 
// 	 break;
//   case SUBSTATUS_DUALDATA:
// 	 move(iRow+4,iCol);
// 	 printw("Computing dual data........");
// 	 break;
//   case SUBSTATUS_DUALDATA_DONE:
// 	 move(iRow+4,iColDone);
// 	 printw("done");
// 	 break;
//   default:
// 	 break;
//   }

//   // Force output
//   refresh();

// #endif

// #ifndef HAVE_CURSES
//   switch (iiSubStatus){
//   case SUBSTATUS_INITTIMESTEPS:
//     cout << "  === Initializing timesteps....." << endl;
//     break;
//   case SUBSTATUS_INITTIMESTEPS_DONE:
//     cout << "  === done" << endl;
//     break;
//   case SUBSTATUS_STEPPING:
//     cout << "  === Stepping..................." << endl;
//     break;
//   case SUBSTATUS_STEPPING_DONE:
//     cout << "  === done" << endl;
//     break;
//   case SUBSTATUS_DUALDATA:
//     cout << "  === Computing dual data........" << endl;
//     break;
//   case SUBSTATUS_DUALDATA_DONE:
//     cout << "  === done" << endl;
//     break;
//   default:
//     break;
//   }
// #endif

//   // Save sub status
//   iSubStatus = iiSubStatus;

// }
// //----------------------------------------------------------------------------
// void Output::SetData(int    iiSizeOfSystem,
// 							int    iiCurrentSolution,
// 							double ddErrorEstimate1,
// 							double ddErrorEstimate2,
// 							double ddErrorEstimate3,
// 							double ddTolerance,
// 							double ddStabilityFactor)
// {
//   // This function prints data that is constant throughout one computation.
  
//   iSizeOfSystem    = iiSizeOfSystem;
//   iCurrentSolution = iiCurrentSolution;
//   dErrorEstimate1  = ddErrorEstimate1;
//   dErrorEstimate2  = ddErrorEstimate2;
//   dErrorEstimate3  = ddErrorEstimate3;
//   dTolerance       = ddTolerance;
//   dStabilityFactor = ddStabilityFactor;

//   // Update the data. Pass saved data for changing variables.
//   // This is so that we may use the same function for printing.
//   // We want to be able to change the "constant" data and not the
//   // changing data when the solution is finished and we compute a new
//   // error estimate.
//   WriteData();
  
// }
// //----------------------------------------------------------------------------
// void Output::Debug(char *cString, int iRow)
// {
//   // This function writes debug messages.

//   // Don't write any output if we shouldn't do that
//   if ( iOutputWhere == ADDRESS_DEVNULL ) return;

// #ifdef HAVE_CURSES
  
//   // Clear previous message
//   for (int i=DEBUG_COL;i<(COLS-1);i++){
//     move(DEBUG_ROW+iRow,i);
// 	 printw(" ");
//   }

//   // Print "Debug: "
//   move(DEBUG_ROW,2);
//   printw("Debug:");

//   // Print the message
//   move(DEBUG_ROW+iRow,DEBUG_COL);
//   printw("%s",cString);

//   // Force output
//   refresh();
// #endif

// #ifndef HAVE_CURSES
//   cout << "Debug: " << cString << endl;
// #endif

// }
// //----------------------------------------------------------------------------
// void Output::DebugValue(char *cString, double dNumber, int iRow)
// {
//   // This function writes debug messages with values.

//   // Don't write any output if we shouldn't do that
//   if ( iOutputWhere == ADDRESS_DEVNULL ) return;
  
// #ifdef HAVE_CURSES
//   // Clear previous message
//   for (int i=DEBUG_COL;i<(COLS-1);i++){
// 	 move(DEBUG_ROW+iRow,i);
// 	 printw(" ");
//   }

//   // Print "Debug: "
//   move(DEBUG_ROW,2);
//   printw("Debug:");

//   // Print the message
//   move(DEBUG_ROW+iRow,DEBUG_COL);
//   printw("%s %e",cString,dNumber);

//   // Force output
//   refresh();
// #endif
 
// #ifndef HAVE_CURSES
//   cout << "Debug: " << cString << dNumber << endl;
// #endif

// }
// //----------------------------------------------------------------------------
// void Output::SetDualSize(double ddSizeOfDualInterval)
// {
//   // This function sets the size of the dual interval.

//   dSizeOfDualInterval = ddSizeOfDualInterval;

//   // Fix size
//   if ( dSizeOfDualInterval > 1.0 )
// 	 dSizeOfDualInterval = 1.0;
//   else if ( dSizeOfDualInterval < 0.0 )
// 	 dSizeOfDualInterval = 0.0;
 
// }
// //----------------------------------------------------------------------------
// //----------------------------------------------------------------------------
// void Output::WriteData()
// {
//   // This function writes some data.

//   // Don't write any output if we shouldn't do that
//   if ( iOutputWhere == ADDRESS_DEVNULL ) return;
  
// #ifdef HAVE_CURSES

//   int j1 = 2;
//   int j2 = COLS/2;
//   int i;
  
//   int iMemory = 0;
//   int iTotalTime = 0;

//   // Left column

//   i = 4;
  
//   move(i++,j1);
//   printw("Size of system: %d      ",iSizeOfSystem);

//   move(i++,j1);
//   printw("Size of slab:   %d      ",iSizeOfSlab);
  
//   move(i++,j1);
//   printw("Size cleared:   %d      ",iSizeCleared);

//   move(i++,j1);
//   printw("Iterations:     %.2f    ",dNumberOfIterations);

//   move(i++,j1);
//   printw("Conservation:   %.0f    ",dMaxConservation);
  
//   i++;
  
//   move(i++,j1);
//   printw("Memory usage:   %d kb   ",iMemory);
  
//   move(i++,j1);
//   printw("Total time:     %d s    ",iTotalTime);  

//   // Right column

//   i = 4;
  
//   move(i++,j2);
//   printw("Stability factor:   %1.3e ",dStabilityFactor);

//   move(i++,j2);
//   printw("Error estimate I:   %1.3e ",dErrorEstimate1);

//   move(i++,j2);
//   printw("Error estimate II:  %1.3e ",dErrorEstimate2);

//   move(i++,j2);
//   printw("Error estimate III: %1.3e ",dErrorEstimate3);

//   move(i++,j2);
//   printw("Tolerance:          %1.3e ",dTolerance);

//   move(i++,j2);
//   printw("Current solution:   %d    ",iCurrentSolution);  
// #endif

// }
// //----------------------------------------------------------------------------
// void Output::WriteLogo()
// {
//   // This function writes the logo

//   // Don't write any output if we shouldn't do that
//   if ( iOutputWhere == ADDRESS_DEVNULL ) return;
  
// #ifdef HAVE_CURSES
//   int iRow = 12;
//   int iCol = COLS/2 + 1;
  
//   move(iRow++,iCol); printw("|  ___                             ");
//   move(iRow++,iCol); printw("|   | _  _  _  _  _   o|  _        ");
//   move(iRow++,iCol); printw("|   |(_|| |(_|(_|| |\\/||<(_| %s",PROGRAM_VERSION);
//   move(iRow++,iCol); printw("|           _|      /              ");
//   move(iRow++,iCol); printw("|  - the multiadaptive ODE-solver  ");
//   move(iRow++,iCol); printw("|                                  ");
//   move(iRow++,iCol); printw("|  Anders Logg                     ");
//   move(iRow++,iCol); printw("|  Chalmers Finite Element Center  ");
//   move(iRow++,iCol); printw("|  Göteborg 2000                   ");

//   // Force output
//   refresh();
// #endif

// #ifndef HAVE_CURSES
//   cout << endl;
//   cout << "|  ___                             " << endl;
//   cout << "|   | _  _  _  _  _   o|  _        " << endl;
//   cout << "|   |(_|| |(_|(_|| |\\/||<(_| " << PROGRAM_VERSION << endl;
//   cout << "|           _|      /              " << endl;
//   cout << "|  - the multiadaptive ODE-solver  " << endl;
//   cout << "|                                  " << endl;
//   cout << "|  Anders Logg                     " << endl;
//   cout << "|  Chalmers Finite Element Center  " << endl;
//   cout << "|  Göteborg 2000                   " << endl; 
//   cout << endl;
// #endif

// }
// //----------------------------------------------------------------------------

