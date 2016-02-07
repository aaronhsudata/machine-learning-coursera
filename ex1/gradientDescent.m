function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

% theta1array = zeros(m,1);
% for i = 1:m
% theta1array(i) = ( dot(theta, X(i,:)) - y(i) ) * X(i,1);
% end
% theta(1) = theta(1) - alpha * (1/m) * sum(theta1array);
% 
% theta2array = zeros(m,1);
% for i = 1:m
% theta2array(i) = ( dot(theta, X(i,:)) - y(i) ) * X(i,2);
% end
% theta(2) = theta(2) - alpha * (1/m) * sum(theta2array);

% why did the below work and the one above didn't? strange...

theta1array = zeros(m,1);
theta2array = zeros(m,1);

for i = 1:m
theta1array(i) = ( dot(theta, X(i,:)) - y(i) ) * X(i,1);
theta2array(i) = ( dot(theta, X(i,:)) - y(i) ) * X(i,2);
end
theta(1) = theta(1) - alpha * (1/m) * sum(theta1array);
theta(2) = theta(2) - alpha * (1/m) * sum(theta2array);



computeCost(X, y, theta);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    

end

end
