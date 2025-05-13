%----------------------------------------------------------------
% File: basic_script.m
%----------------------------------------------------------------
%
% Author: Marek Rychlik (rychlik@arizona.edu)
% Date: Mon Apr 7 14:19:32 2025
% Copying: (C) Marek Rychlik , 2020. All rights reserved.
%
%----------------------------------------------------------------
% Basic operations on raw images
% Source image: https://www.reddit.com/r/EditMyRaw/comments/1jt4ecw the_official_weekly_raw_editing_challenge/

% Updated image path
raw_img_filename = fullfile('C:', 'Users', 'jmcoh', 'Documents', ...
    'Arizona', '589', 'Sp25', '589BFinalProject', 'DSC00099.ARW');

% Set up tiled layout
t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
linked_axes = [];

% Read raw data and info
cfa = rawread(raw_img_filename);
ax = nexttile; imagesc(bitand(cfa, 7)); title('xor with 7');
linked_axes = [linked_axes, ax];

info = rawinfo(raw_img_filename);
disp(info);

% Extract color planes
Iplanar = raw2planar(cfa);
planes = 'rggb';
ang = info.ImageSizeInfo.ImageRotation;

for j = 1:size(Iplanar, 3)
    ax = nexttile;
    imagesc(imrotate(Iplanar(:,:,j), ang));
    title(['Plane ', num2str(j), ': ', planes(j)]);
    colorbar; colormap gray;
    linked_axes = [linked_axes, ax];
end

% Demosaicing and color
Irgb = raw2rgb(raw_img_filename);
ax = nexttile;
image(Irgb);  % No resizing or scaling
title('Demosaiced RGB image');
colorbar;
linked_axes = [linked_axes, ax];

% --- Apply ROF smoothing ---
gray_img = rgb2gray(Irgb);                  % Convert to grayscale
gray_img = double(gray_img);               % Force single precision, no scaling

lambda = 0.1;
epsilon = 1e-2;
smoothed = smooth_image_rof(gray_img, lambda, epsilon);

% --- Gradient descent ROF smoothing ---
u_grad = rof_gradient_descent(gray_img, lambda, epsilon);
u_grad = double(u_grad);  % Match class with smoothed

% Compare numerically
mse_val = immse(smoothed, u_grad);
psnr_val = psnr(smoothed, u_grad);

fprintf('MSE between smooth_image_rof and gradient descent: %.6f\n', mse_val);
fprintf('PSNR between smooth_image_rof and gradient descent: %.2f dB\n', psnr_val);

% Display ROF result
ax = nexttile;
imagesc(smoothed);
title('ROF Smoothed Grayscale');
colorbar; colormap gray;
linked_axes = [linked_axes, ax];

% Display Gradient Descent result
ax = nexttile;
imagesc(u_grad);
title('Gradient Descent ROF');
colorbar; colormap gray;
linked_axes = [linked_axes, ax];

% Link axes
linkaxes(linked_axes);
