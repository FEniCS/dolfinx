// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "Terminal.hh"
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

//-----------------------------------------------------------------------------
void Terminal::Status(int level, const char *format, ...)
{
  switch (level){
  case 0:
	 printf("  ");
	 break;
  case 1:
	 printf("    ");
	 break;
  case 2:
	 printf("      ");
	 break;
  default:
	 printf("        ");
  }

  va_list aptr;
  va_start(aptr,format);

  vprintf(format,aptr);
  printf("\n");
  
  va_end(aptr);  
}
//-----------------------------------------------------------------------------
void Terminal::Message(int level, const char *format, ...)
{
  if ( (level > debug_level) && (debug_level >= 0) )
	 return;
  
  va_list aptr;
  va_start(aptr,format);
  
  printf("  ");
  vprintf(format,aptr);
  printf("\n");
  
  va_end(aptr);  
}
//-----------------------------------------------------------------------------
void Terminal::Progress(int level, double progress, const char *format, ...)
{
  int N = DOLFIN_TERM_WIDTH - 15;
  int n = ( (int) ( progress*( (double) N) ));

  // Print a description
  va_list aptr;
  va_start(aptr,format);

  char description[DOLFIN_LINELENGTH];
  vsprintf(description,format,aptr);
  
  va_end(aptr);  

  printf("| %s",description);
  int length = StringLength(description);
  for (int i=0;i<(N-length-1);i++)
	 printf(" ");
  printf("|\n");
  
  // Print the progress bar
  if ( progress < 0.0 )
	 progress = 0.0;
  if ( progress > 1.0 )
	 progress = 1.0;
  
  printf("|",level);
  for (int i=0;i<n;i++)
	 printf("=");
  if ( (n>0) && (n<N) ){
	 printf("|");
	 n += 1;
  }
  for (int i=n;i<N;i++)
	 printf("-");
  printf("| %.1f\%\n",100.0*progress);
}
//-----------------------------------------------------------------------------
void Terminal::Regress(int level, double progress, double maximum,
							  const char *format, ...)
{
  int N = DOLFIN_TERM_WIDTH - 15;
  int n = ( (int) ( progress*( (double) N) ));

  // Print a description
  va_list aptr;
  va_start(aptr,format);

  char description[DOLFIN_LINELENGTH];
  vsprintf(description,format,aptr);
  
  va_end(aptr);  

  printf("| %s",description);
  int length = StringLength(description);
  for (int i=0;i<(N-length-1);i++)
	 printf(" ");
  printf("|\n");
  
  // Print the progress bar
  if ( progress < 0.0 )
	 progress = 0.0;
  if ( progress > 1.0 )
	 progress = 1.0;
  
  printf("|",level);
  for (int i=0;i<n;i++)
	 printf("=");
  if ( (n>0) && (n<N) ){
	 printf("|");
	 n += 1;
  }
  for (int i=n;i<N;i++)
	 printf("-");
  printf("| %.1f\%\n",100.0*progress);
}
//-----------------------------------------------------------------------------
void Terminal::Value(const char *name, Type type, ...)
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
	 printf("  %s = %f\n",name,val_double);
	 break;
  case type_real:
	 val_double = va_arg(aptr,double);
	 printf("  %s = %f\n",name,val_double);
	 break;
  case type_int:
	 val_int = va_arg(aptr,int);
	 printf("  %s = %d\n",name,val_int);
	 break;
  case type_bool:
	 val_bool = va_arg(aptr,int);
	 printf("  %s = %s\n",name,(val_bool ? "true" : "false"));
	 break;
  case type_string:
	 val_string = va_arg(aptr,char *);
	 printf("  %s = \"%s\"\n",name,val_string);
	 break;
  default:
	 InternalError("Terminal:Value","Unknown type.");
  }
  
  va_end(aptr);  
}
//-----------------------------------------------------------------------------
void Terminal::Warning(const char *format, ...)
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
void Terminal::Error(const char *format, ...)
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
void Terminal::InternalError(const char *function, const char *format, ...)
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
