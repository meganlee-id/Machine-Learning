function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p); % numel means number of elements

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
%~~~~~ input: ~~~~~~
% disp(size(X)) % 3 X 1
% disp(size(p)) % 1 X 1
%~~~~~ output: ~~~~~~
% disp(size(X_poly)) % 3 X 8

X_poly_acc = [];
for i=1:p
    ith_col = X .^ i;
    X_poly_acc = [X_poly_acc;ith_col'];
end

X_poly = X_poly_acc';


% =========================================================================

end
