function idx = findClosestCentroids(X, centroids)
% findClosestCentroids computes the centroid memberships for every example
%   idx = findClosestCentroids (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% X         [15, 11]
% centroid  [5,  11]
% K          5
% each row in X has 11 features, total of 15 data points (rows) in X
% each row in centroid has 11 features, total of 5 centroid (num of clusters)

K = size(centroids, 1);
num_rows_X = size(X, 1);

% You need to return the following variables correctly.
idx = zeros(num_rows_X, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


for row = 1 : num_rows_X % for each data point in X
    min_dist_sq = 1000000;  % min dist for tbe current row in X
    for k = 1 : K;
        row_X = X(row, :);
        row_centroid = centroids(k, :);
        row_diff = row_X - row_centroid;
        dist_sq = row_diff * row_diff';
        if dist_sq < min_dist_sq
            idx(row) = k;
            min_dist_sq = dist_sq;
        end
end


% =============================================================

end

