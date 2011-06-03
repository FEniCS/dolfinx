% Copyright (C) 2003-2005 Anders Logg
% Copyright (C) 2003 Johan Hoffman
%
% This file is part of DOLFIN.
%
% DOLFIN is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% DOLFIN is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU Lesser General Public License for more details.
%
% You should have received a copy of the GNU Lesser General Public License
% along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
%
% First added:  2003-11-07
% Last changed: 2005

function M = plotslab(debugfile, interval, drawupdates, drawtext, saveps, savepng)

% Usage: M = plotslab(debugfile, interval, drawupdates, drawtext, saveps, savepng)
%
% Draws and creates a movie M of a time slab. A debug file
% is created by DOLFIN when the option 'debug time steps'
% is specified.
%
% Arguments:
%
%   debugfile   - name of file containing time stepping data
%   interval    - plot elements within interval
%   drawupdates - blink element updates
%   drawtext    - draw extra text
%   saveps      - save a postscript file for each frame
%   savepng     - save a png file for each frame

% Load the steps
disp('Loading file...')
timesteps = load(debugfile);

% Get the number of components
N  = max(timesteps(:,2)) + 1;

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
for i = 1:size(timesteps,1)

  a  = timesteps(i,1);
  n  = timesteps(i,2);
  t1 = timesteps(i,3);
  t2 = timesteps(i,4); 

  if t2 >= T1
    break;
  end

end

% Draw the elements
for j = i:size(timesteps, 1)

  % Get next element
  a  = timesteps(j,1);
  n  = timesteps(j,2);
  t1 = timesteps(j,3);
  t2 = timesteps(j,4);

  % Go to next element if we don't want to draw updates
  if ~drawupdates & a == 1
    continue
  end
  
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
      if saveps
	print('-depsc', ['frame_' sprintf('%.4d', framecount) '.eps'])
      elseif savepng
	print('-dpng', '-r0', ['frame_' sprintf('%.4d', framecount) '.png'])
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
  if saveps
    print('-depsc', ['frame_' sprintf('%.4d', framecount) '.eps'])
  elseif savepng
    print('-dpng', '-r0', ['frame_' sprintf('%.4d', framecount) '.png'])
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
