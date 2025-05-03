clear all; close all; clc;

%% function definition (Rosenbrock fuction)

fun = @(x, y) ((x - 1).^2 + (y - x.^2).^2);

%% initialize start 

init =  [0.5, 0.75];


%% gradient call

epsilon = 1e-5;

dfx = @(x, y) (fun(x + epsilon, y) - fun(x, y)) / epsilon;


dfy = @(x, y) (((fun(x , y + epsilon) - fun(x, y)) / epsilon));

%% gradient descent

function [x_star, y_star, path] = gradient_descent_2d(f, dfx, dfy, x_init, y_init, iter)

    if nargin < 6 || isempty(iter)
    
        iter = 1000;
    
    end

    curr = [x_init, y_init];

    eta = 0.01;

    path = zeros(iter, 2);

    epsilon = 1e-10;

    for i = 1:iter

        x = curr(1);

        y = curr(2);

        gradient_eval = [dfx(x, y), dfy(x, y)];

        disp(gradient_eval)

        if abs(gradient_eval) < epsilon
           
            break;

        end

        % step
        
        curr = curr - eta .* gradient_eval;

        path(i, :) = curr;

        
    end

x_star = curr(1);

y_star = curr(2);

end



[x_star, y_star, path] = gradient_descent_2d(fun, dfx, dfy, init(1), init(2));