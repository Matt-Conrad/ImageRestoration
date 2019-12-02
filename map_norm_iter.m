function [out_image,residual] = map_norm_iter(measured_image, start_image, H, num_iters, mean_image, beta)

% function outimage = map_norm_iter(measured_image, start_image, H,
% num_iters, mean_image, beta)

% Determines iterative MAP solution by the method of steepest
% ascent.  The measured image and starting images are specified.  The 
% H_matrix is the explicit linear transfer mapping from input to output.  
% num_iters gives the number of iterations.  The prior is assumed to be
% normally distributed, with a given mean.  Beta determines the relative
% weight placed on the prior versus the data.

% initialize
current_estimate = start_image;
residual = zeros(num_iters+1,1);

% Iterate
for i = 1:num_iters
    
    % Compute current residual error image
    error = measured_image - H * current_estimate(:);
    
    % Evaluate the squared error
    residual(i) = error' * error + beta * (current_estimate(:) - mean_image(:))' * (current_estimate(:) - mean_image(:));
    
    % Compute the gradient direction for the MAP objective function
    h = H' * error - beta .* (current_estimate(:) - mean_image(:));
    
    % Compute the step size
    numerator =  h' * H' * error - beta .* h' * (current_estimate(:) - mean_image(:));
    temp = H * h;
    denominator = temp' * temp + beta * h' * h;
    t = numerator / denominator;
    
    % Apply the update
    current_estimate = current_estimate + t .* h;
    
end

% Compute final residual
error = measured_image - H * current_estimate(:);
residual(i+1) = error' * error + beta * (current_estimate(:) - mean_image(:))' * (current_estimate(:) - mean_image(:));;

out_image = current_estimate;

return
