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

    f = double(f);
    [H, W] = size(f);
    K = numel(lambda);
    L = numel(epsilon);
    msd = zeros(K, L);

    bytes_per_element = 8;  % for double precision
    estimated_bytes = H * W * bytes_per_element;

    for i = 1:K
        for j = 1:L
            % Check GPU availability and memory safety
            use_gpu = (gpuDeviceCount > 0) && estimated_bytes < 7.5e9;
            if use_gpu
                f = gpuArray(f);
            end

            % Compute ROF-smoothed result for this (lambda, epsilon) pair
            u = smooth_image_rof(f, lambda(i), epsilon(j));

            % Ensure it's on CPU for comparison
            u = gather(u);
            diff = u - f;
            msd(i, j) = mean(diff(:).^2);
         end
     end
    % f = double(f);  % Ensure single precision
    % 
    % % Run the ROF smoothing algorithm
    % u = smooth_image_rof(f, lambda, epsilon);  % u is H x W x K x L
    % 
    % [H, W] = size(f);
    % K = numel(lambda);
    % L = numel(epsilon);
    % 
    % % Expand f to match u size
    % f_rep = repmat(f, 1, 1, K, L);
    % 
    % % Compute squared difference
    % diff = u - f_rep;
    % diff_sq = diff .^ 2;
    % 
    % % Compute mean square difference
    % msd = sum(diff_sq, [1 2]) / (H * W);
    % msd = reshape(msd, K, L);   % Ensure 2D output shape
    % msd = double(msd);          % Ensure single precision output
end