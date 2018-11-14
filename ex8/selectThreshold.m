function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).


%-------- input size ------------%
% disp(max(pval));  % 1
% disp(min(pval));  % 0.001
% disp(size(pval)); % 14 X 1 -- 14 rows/data points, the predicted value/possibility for each row between 0 and 1
% disp(size(yval)); % 14 X 1 -- 14 rows/data points, the know value for each row, value could either be 1 or 0

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;


stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

    % ====================== YOUR CODE HERE ======================
    % Instructions: the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions


    % get true positive, false positive, false negative
    predicted_val = (pval < epsilon);
    tp = sum((predicted_val == 1) & (yval == 1));
    fp = sum((predicted_val == 1) & (yval == 0));
    fn = sum((predicted_val == 0) & (yval == 1));

    % precision
    precision = 0;
    if tp + fp > 0
      precision = tp / (tp + fp);
    end

    % recall
    recall = 0;
    if (tp + fn) > 0
      recall = tp / (tp + fn);
    end

    % f1 score
    f1 = 0;
    if precision + recall > 0
      F1 = 2 * (precision * recall) / (precision + recall);
    end


    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end