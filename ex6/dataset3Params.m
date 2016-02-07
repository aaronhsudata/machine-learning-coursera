function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Define the C's and sigma's that we wish to test
CList = [.01, .03, .1, .3, 1, 3, 10, 30];
sigmaList = CList;

% Initialize error to be large
bestError = 10000000000;

% Loop through all C's and all sigma's
for CTest = CList
  for sigmaTest = sigmaList
  
    % Calculate the SVM for given parameters
    model = svmTrain(X, y, CTest, @(x1, x2) gaussianKernel(x1, x2, sigmaTest));
    predictions = svmPredict(model, Xval);
    
    % Calculate the prediction error
    error = mean(double(predictions ~= yval))
    
    % If prediction error is lower than the best error so far, keep parameters
    if error < bestError
      bestError = error;
      C = CTest
      sigma = sigmaTest
      bestError
    end
  end
end  





% =========================================================================

end
