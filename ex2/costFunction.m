function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Loop across thetas
for j = 1:size(theta)
    
    for i = 1:m % run through training examples
        %   Calculate h with given theta vector and training ex
        h(i) = sigmoid(dot(theta, X(i,:))) ;
        %   Calculate the rest of the term in sigma for that training ex
        array(i) = ( h(i) - y(i) ) * X(i,j);
    end
    
    % The grad of J wrt theta_j
    grad(j) = (1/m) * sum(array); %%% not array(j)!!!!!
end

for i = 1:m
term(i) = (-1 * y(i)) * log(sigmoid(dot(theta,X(i,:)))) - ...
    (1 - y(i)) * (log(1 - sigmoid(dot(theta,X(i,:)))));
end
J = (1/m) * sum(term);





% =============================================================

end
