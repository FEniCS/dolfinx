// Copyright (C) 2002 [fill in name]
// Licensed under the GNU GPL Version 2.

#include "ProblemTemplate.hh"
#include "EquationTemplate.hh"

//-----------------------------------------------------------------------------
const char *ProblemTemplate::Description()
{
	return "My new equation";
}
//-----------------------------------------------------------------------------
void ProblemTemplate::Solve()
{
	display->Message(0,"Solving...");	
}
//-----------------------------------------------------------------------------
