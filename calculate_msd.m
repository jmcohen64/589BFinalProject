function msd = calculate_msd(f, lambda, epsilon)
% CALCULATE_MSD - returns MSD for a given degraded image and ROF parameters
% MSD = CALCULATE_MSD(F, LAMBDA , EPSILON) - find MSD
% Arguments:
% F - the degraded image
% LAMBDA - the 'smoothing ' parameter (scalar or vector)
% EPSILON - the 'regularization ' parameter (scalar or vector)
% Returns:
% MSD - the MSD of the degraded image (scalar or 1D array)
% If LAMBDA or EPSILON is a vector , the result should be an array of size
% K-by-L, where and K and L are the lengths of LAMBDA and EPSILON , respectively.
