#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <dolfin/constants.h>
#include <dolfin/meminfo.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::meminfo()
{
  // Get process id
  int pid = (int) getpid();

  // Read values
  char command[DOLFIN_WORDLENGTH];

  system("/bin/rm -f meminfo.tmp");

  sprintf(command, "echo \"- Memory usage for process %d\" >> meminfo.tmp", pid);
  system(command);
  
  sprintf(command, "cat /proc/%d/status | grep VmSize | /usr/bin/awk '{print \"- Size:     \"$2\" \"$3}' >> meminfo.tmp", pid);
  system(command);
  
  sprintf(command, "cat /proc/%d/status | grep VmRSS  | /usr/bin/awk '{print \"- Resident: \"$2\" \"$3}' >> meminfo.tmp", pid);
  system(command);

  system("cat meminfo.tmp");
}
//-----------------------------------------------------------------------------
