function u = smooth_image_rof(f, lambda, epsilon)
%SMOOTH_IMAGE_ROF Restores an image using the ROF model over parameter grid.
%   U = SMOOTH_IMAGE_ROF(F, LAMBDA, EPSILON) computes ROF-denoised versions of
%   image F over all combinations of lambda and epsilon.
%
%   - f: 2D degraded image (H x W).
%   - lambda, epsilon: vectors of parameters.
%   - u: 4D output of shape H x W x K x L (K = length(lambda), L = length(epsilon)).

    useGPU = (gpuDeviceCount > 0);
    if useGPU
        f = gpuArray(f);
    end

    [H, W] = size(f);
    lambda = lambda(:);  
    epsilon = epsilon(:);
    K = length(lambda);  
    L = length(epsilon);
    eps2_vals = epsilon.^2;

    uCell = cell(K * L, 1);
    max_iter = 100;
    tol = 1e-4;

    parfor idx = 1:(K * L)
        [k, l] = ind2sub([K, L], idx);
        lam = lambda(k);  
        eps2 = eps2_vals(l);
        uk = f;

        for iter = 1:max_iter
            up = padarray(uk, [1 1], 'symmetric');
            ux = up(2:end-1,3:end) - up(2:end-1,2:end-1);
            uy = up(3:end,2:end-1) - up(2:end-1,2:end-1);
            mag = sqrt(eps2 + ux.^2 + uy.^2);
            px = ux ./ mag;
            py = uy ./ mag;

            pxp = padarray(px, [0 1], 'pre');
            pyp = padarray(py, [1 0], 'pre');
            div = pxp(:,2:end) - pxp(:,1:end-1) + pyp(2:end,:) - pyp(1:end-1,:);

            unew = f - lam * div;

            rel_change = norm(unew(:) - uk(:)) / (norm(uk(:)) + eps);
            if rel_change < tol
                break;
            end
            uk = unew;
        end

        % Only gather if GPU was used
        if useGPU
            uk = gather(uk);
        end
        uCell{idx} = uk;
    end

    % Reassemble into 4D array
    u = zeros(H, W, K, L, 'like', f);
    for idx = 1:(K * L)
        [k, l] = ind2sub([K, L], idx);
        u(:,:,k,l) = uCell{idx};
    end

    if useGPU
        u = gather(u);
    end
end
