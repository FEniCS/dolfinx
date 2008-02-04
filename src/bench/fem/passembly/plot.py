from pylab import *

if (size(sys.argv) < 3):
  print 'Usage:', sys.argv[0], '<title> <plotfile> [plotfile 2 ... plotfile n]'
  print """title - The plot title
           plotfile - Output file from the benchmark program
        """
  sys.exit(1)

plottitle = sys.argv[1]

for filename in sys.argv[2:]:
  file = open(filename, 'r')
  lines = file.readlines();
  linelabel = lines[0]
  axis = lines[1].split()
  (xname, yname) = (axis[0], axis[1])
  xarr = [(int)(lines[2].split()[0])]
  yarr = [lines[2].split()[1]]
  for line in lines[3:]:
    x = (int)(line.split()[0])
    y = line.split()[1]
    if x == xarr[-1]:
      if y > yarr[-1]:
        yarr[-1] = y
    else:
      xarr.append(x)
      yarr.append(y)

  #loglog(xarr, yarr, '-o', label=linelabel)
  plot(xarr, yarr, '-o', label=linelabel)
  legend(loc='best')
  file.close()

title(plottitle)
xlabel(xname)
ylabel(yname)
show()
