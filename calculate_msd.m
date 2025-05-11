function msd = calculate_msd(f, lambda, epsilon)
% CALCULATE_MSD - Computes Mean Square Difference between f and denoised result(s)
%
% Inputs:
%   f       - degraded image (HxW), single precision
%   lambda  - regularization parameter(s), scalar or vector
%   epsilon - smoothing parameter(s), scalar or vector
%
% Output:
%   msd     - KxL matrix, where K = numel(lambda), L = numel(epsilon)

    f = double(f);  % Ensure single precision

    % Run the ROF smoothing algorithm
    u = smooth_image_rof(f, lambda, epsilon);  % u is H x W x K x L

    [H, W] = size(f);
    K = numel(lambda);
    L = numel(epsilon);

    % Expand f to match u size
    f_rep = repmat(f, 1, 1, K, L);

    % Compute squared difference
    diff = u - f_rep;
    diff_sq = diff .^ 2;

    % Compute mean square difference
    msd = sum(diff_sq, [1 2]) / (H * W);
    msd = reshape(msd, K, L);   % Ensure 2D output shape
    msd = double(msd);          % Ensure single precision output
end