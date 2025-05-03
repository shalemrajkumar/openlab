clear all; close all; clc;



%% objective function

f = @(x) (x-7).^2 + 1;

%% plot obj function

x = linspace(-10, 10, 1000);

plot(x, f(x))

%% golden section search

function [a, b] = golden_section_search(a, b, f, epsilon) 

    if nargin < 4 || isempty(epsilon)
        epsilon = 1e-6;
    end

    gr = (sqrt(5) - 1) / 2;

    x1 = a + gr*(b-a);
    x2 = b - gr*(b-a);

    for i = 1:1000

        if b-a < epsilon
            break;
        end 

        if f(x1) < f(x2) % since x1 is close to minima dump the other 1 - psi interval st minima is always lies inside the interval

            a = x2;

            x2 = x1;

            x1 = a + gr*(b-a);

        else

            b = x1;

            x1 = x2;

            x2 = b - gr*(b-a);

        end

    end

end

%% testing the funtion


[a, b] = golden_section_search(-10, 10, f, 1e-5);

disp([a, b])