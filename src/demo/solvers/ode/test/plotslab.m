function M = plotslab(interval, drawtext, saveps)

% Usage: M = plotslab(interval, drawtext, saveps)
%
% Draws and creates a movie M of a time slab. Data
% is taken from the file 'timeslab.debug' which is
% created by DOLFIN when the option 'debug time slab'
% is specified.
%
% Arguments:
%
%   interval - plot elements within interval
%   drawtext - draw extra text
%   saveps   - save a postscript file for every frame
%
% Copyright (C) 2003 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.

% Load the steps
disp('Loading file...')
load('timeslab.debug');

% Get the number of components
N  = max(timeslab(:,2)) + 1;

% Get the time interval
T1 = interval(1);
T2 = interval(2);

% Create figure
clf
axis([T1 T2 -0.3*N 1.2*N])
axis off
hold on
if drawtext
  h1 = text(T1,1.1*N,'Action:');
  h2 = text(T1,-0.1*N,['t = ' num2str(T1)]);
  h3 = text(T1,-0.15*N,['i = ' num2str(0)]);
end
plot([T1 T2],[0 0],'k');
framecount = 1;
iterations = 0;

% Clear the movie
clear M

% Step to the correct position
for i = 1:size(timeslab,1)

  a  = timeslab(i,1);
  n  = timeslab(i,2);
  t1 = timeslab(i,3);
  t2 = timeslab(i,4); 

  if t2 >= T1
    break;
  end

end

% Draw the elements
for j = i:size(timeslab, 1)

  % Get next element
  a  = timeslab(j,1);
  n  = timeslab(j,2);
  t1 = timeslab(j,3);
  t2 = timeslab(j,4);
  
  % Check if we have finished
  if t1 > T2
    break
  end

  % Check action
  switch a
    case 0
      
      % Draw a new box
      %c = [128 156 178]/256;
      c = 'b';
      drawbox([t1 n], [t2 (n+1)], c);

      % Draw text
      if ( drawtext )
	set(h1, 'String', 'Action: Creating new element')
	set(h2, 'String', ['t = ' num2str(t1)])
	set(h3, 'String', ['i = ' num2str(n)])
      end

    case 1

      % Blink box
      %c = [0 98 178]/256;
      c = 'r';
      cc = [250 130 180]/256;
      drawbox([t1 n], [t2 (n+1)], c);

      % Draw text
      if ( drawtext )
	set(h1, 'String', 'Action: Updating element')
	set(h2, 'String', ['t = ' num2str(t1)])
	set(h3, 'String', ['i = ' num2str(n)])
      end
      
      % Save frame
      drawnow
      M(framecount) = getframe;
      if ( saveps )
	print('-depsc',['frame_' num2str(framecount) '.eps'])
      else
	disp('Press any key to continue')
	pause
      end		  
      framecount = framecount + 1;

      % Restore box
      %c = [128 156 178]/256;
      c = 'b';
      drawbox([t1 n], [t2 (n+1)], c);

  end  
  
  % Save frame
  drawnow
  M(framecount) = getframe;
  if ( saveps )
    print('-depsc',['frame_' num2str(framecount) '.eps'])
  else
    disp('Press any key to continue')
    pause
  end		  
  framecount = framecount + 1;
  
end

function drawbox(x, y, c)

% This function draws a box from x to y with color c

x1 = [x(1) y(1) y(1) x(1)];
x2 = [x(2) x(2) y(2) y(2)];

fill(x1, x2, c)
