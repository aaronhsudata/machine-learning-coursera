function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X = [ones(m,1) X];

J = (1/(2*m)) * sum(((X * theta) - y).^2)...
    + lambda/(2*m) * sum(theta(2:size(theta,1)).^2);
% need the paren around 2*m in first line!

grad = zeros(size(theta,2));
grad(1) = (1/m) * sum(((X * theta) - y) .* X(:,1));
(size(X,2)-1);
for k = 1:(size(X,2)-1)
    grad(k+1) = (1/m) * sum(((X * theta) - y) .* X(:,k+1)) + (lambda/m) * theta(k+1); %k+1
end

% grad1 = (1/m) * sum(((X * theta) - y) .* X(:,2)) + (lambda/m) * theta(2)

% grad = [grad0; grad1]
% 
% size(X)
% size(theta)


% =========================================================================

grad = grad(:);

end
