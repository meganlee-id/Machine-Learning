function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% disp(size(X));    15 X 11  15 rows, each row has 11 features/dims
% disp(size(idx));  15 X 1   15 rows, each row indicate which cluster row in X belongs to
% disp(size(K));     1 X 1   1 num, num of clusters
% centroids          K X N   the new centroids for K clusters

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

A = {};         % A{k} holds all data points for idx k
for k = 1 : K   % initialize A. otherwise will have out of boundary error
    A{k} = [];
end

% for each data point in X, assign it to the corresponding cluster A{k}
for row = 1 : m 
    A{idx(row)} = [A{idx(row)}; X(row, :)]; % append current data point to the cluster
end

% for each cluster, calculate it's new centroid
for k = 1 : K
    centroids(k, :) = mean(A{k});
end


% =============================================================

end

