clear all; close all; clc;

% note provide guess S.T f(x1) and f(x2) should be in opposite sign
x_sol = fzero(@(x) fun4bisec(x), [1, 4])

% here we don't know how many roots are there for the given f(x) so just
% scan all the space and look out for sign changes

%Scan the interval and Detect sign changes:

x = linspace(1e-5, 4, 1000);  % Define an interval [a, b]
y = fun4bisec(x);                  % Evaluate your function f at these points
plot(x, y)
grid on


%Apply fzero to each sub-interval:

x_sol = fzero(@(x) fun4bisec(x), [1e-5, 1])


%% function 
function f_name = fun4bisec(x)
    f_name = 2 - x + log(x);
end