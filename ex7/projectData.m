function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% input / output sizes
% disp(size(X));  % 15 X 11 -- 15 rows/data points, each has 11 features
% disp(size(U));  % 11 X 11 -- 11 rows, if we take first k row, we could reduce dim to k
% disp(K);        % 5       -- total 5 clusters
% disp(Z);        % 15 X 5  -- 15 rows/data points, each has K features reduced dims

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% vectors with k dim, get first k columns
k_vector = U(:, 1 : K);
% use k_vector to project X into lower dimensions
Z = X * k_vector;

% =============================================================

end
