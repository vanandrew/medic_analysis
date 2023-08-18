function [r p] = paircorr_mod_diag(a,b)
%PAIRCORR Computes on diagonal Pearson's linear correlation coefficient between 
% two matrices with optional significance. Returns r, a 1-by-p1 matrix containing the
% correlation coefficient between each pair of columns in the
% n-by-p1 and n-by-p1 matrices a and b. r is calculated as the dot
% product between two vectors divided by the product of their magnitudes.
% If a second output argument is provided, like so:
% [r p] = paircorr(a,b), then p is the two-tailed significance.
% Computes correlation based on geometric definition of correlation, TOL
% 2011
% TOL, 05/17/22

if nargin<2
    b = a;
end

a = bsxfun(@minus, a, mean(a));
b = bsxfun(@minus, b, mean(b));

mag_a = sqrt(sum(a.^2, 1));
mag_b = sqrt(sum(b.^2, 1));

numer = sum(a .* b,1);

denom = sum(mag_a .* mag_b,1);

r = numer ./ denom;

if nargout > 1
    [n p1] = size(a);
    
    % calculate t-statistic
    t = r ./ sqrt((1 - r.^2)/(n - 2));
    % calculate significance, two-tailed
    p = 2 * tcdf(-abs(t), n - 2);
end

