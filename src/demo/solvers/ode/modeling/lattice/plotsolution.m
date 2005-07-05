% Copyright (C) 2004-2005 Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% First added:  2004-04-08
% Last changed: 2005

primal

n = size(u,1)/4;
p = round(1/2 + sqrt(n/2 - 1/4));
M = size(u,2);

figure(1)
clf

% Lower left large particle
x1 = u(1:2,:);

% Upper right large particle
x2 = u(2*p^2-1:2*p^2,:);

% Lower left small particle
x3 = u(2*p^2+1:2*p^2+2,:);

% Diameter of the lattice
D = sqrt(sum((x1 - x2).^2,1));

% Distance between small and large particle
d = sqrt(sum((x1 - x3).^2,1));

subplot(2,1,1)
plot(t, D)
ylabel('D')
subplot(2,1,2)
plot(t, d)
ylabel('d')
xlabel('t')

figure(2)

for m = 1:M

  clf
  
  for i=1:n
    
    x = u(2*(i-1) + 1, m);
    y = u(2*(i-1) + 2, m);

    if ( i <= p*p )
      plot(x, y, 'o')
    else
      plot(x, y, 'x')
    end
      
    hold on
    
  end

  axis([-0.1 1.1 -0.1 1.1])
  axis equal
  
  disp(['t = ' num2str(t(m)) ' (Press any key to continue)'])
  pause
  
end

