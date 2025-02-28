u = [];
u(1) = 0;
for i=1: 100
    u(end+1) = f(0, u(end));

end

plot(u)

%% function

function result = f(t, u)
    result = u + 0.1*(-3*u^2 + 6);
end