% Copyright (C) 2003 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% Generate plots from the performance tests.

% Non-stiff, M = 100
timings_M100_b0

figure(1)
clf
plot(n, t1,'--o')
hold on
plot(n, t2,'-o')
grid on
xlabel('n')
ylabel('t')
title('multi-adaptive non-stiff, M = 100')

% Non-stiff, M = 200
timings_M200_b0

figure(2)
clf
plot(n, t1,'--o')
hold on
plot(n, t2,'-o')
grid on
xlabel('n')
ylabel('t')
title('multi-adaptive non-stiff, M = 200')

% Stiff, M = 100
timings_M100_b100

figure(3)
clf
plot(n, t1,'--o')
hold on
plot(n, t2,'-o')
grid on
xlabel('n')
ylabel('t')
title('multi-adaptive stiff, M = 100')

% Stiff, M = 200
timings_M200_b100

figure(4)
clf
plot(n, t1,'--o')
hold on
plot(n, t2,'-o')
grid on
xlabel('n')
ylabel('t')
title('multi-adaptive stiff, M = 200')
