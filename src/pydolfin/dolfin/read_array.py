from Numeric import *
from scipy import *

def read_array(fp):
    """Return an array from ASCII-formatted data in the file object
    fp.  This is a less general reimplementation of the function
    scipy.io.read_array(), which has too much overhead (a factor 20
    slower than a simple implementation)."""

    N = 0
    M = 0
    
    while 1:
        line = fp.readline()
        if not line: break
        if(N == 0):
            elements = line.split()
            M = len(elements)
        N += 1
                
    print "N: " + str(N)
    print "M: " + str(M)
                
    fp.seek(0)
                            
    if(M == 1):
        A = zeros((N), 'd')
    elif(N == 1):
        A = zeros((M), 'd')
    else:
        A = zeros((N, M), 'd')

    i = 0
    while 1:
        line = fp.readline()
        if not line: break

        elements = line.split()
        j = 0
        for e in elements:
            if(M == 1):
                A[i] = float(e)
            elif(M == 1):
                A[j] = float(e)
            else:
                A[i, j] = float(e)
            j += 1
        i += 1

    return A
