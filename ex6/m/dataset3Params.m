function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

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

% Vectors of C and sigma
C_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

length_C_vec = length(C_vec);
length_sigma_vec = length(sigma_vec);

predict_error_min = 1000.0;
for ic = 1:length_C_vec
    C_selected = C_vec(ic);
    for is = 1:length_sigma_vec
        sigma_selected = sigma_vec(is);

        model= svmTrain(X, y, C_selected, ...
            @(x1, x2) gaussianKernel(x1, x2, sigma_selected)); 
        predictions = svmPredict(model, Xval);
        predict_error = mean(double(predictions ~= yval));
        if predict_error < predict_error_min
            predict_error_min = predict_error;
            C = C_selected;
            sigma = sigma_selected;
        end  

    end
end

% =========================================================================

end
