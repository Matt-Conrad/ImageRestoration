function out_array = gaussian_blur(in_array, sigma_x, sigma_y, sigma_z, size_of_kernel)
%GAUSSIAN_BLUR Applies a Gaussian blur to an intensity image array
%   Takes a 2D or 3D intensity image array called in_array and applies a
%   Gaussian blur to it by convolving it with a 2D or 3D Gaussian kernel.
%   The Gaussian function uses the double sigma parameters given (in
%   units of pixels), and the size of the kernel is given by the
%   parameter size_of_kernel.  (That is, if N = size_of_kernel, then the
%   kernel array is NxNxN or NxN.)  The output image is
%   out_array and has the same size as the input image.

% Determine the dimensionality of in_array
nDimensions = ndims(in_array);

% Switch for different dimensions
switch (nDimensions)
    case 2
        disp('Input array: 2 dimensions');
        % Create a Gaussian kernel
        kernel = zeros(size_of_kernel,size_of_kernel);
        if mod(size_of_kernel,2) == 0 %case where kernel size is even
            center_of_kernel = size_of_kernel/2 + 0.5;
        else %case where kernel size is odd
            center_of_kernel = ceil(size_of_kernel/2);
        end

        for i = 1:size_of_kernel
            for j = 1:size_of_kernel
                kernel(j,i) = exp(-.5 * (((i-center_of_kernel)/sigma_x)^2 + ((j-center_of_kernel)/sigma_y)^2));
            end
        end
        % Normalize kernel
        kernel = kernel/sum(sum(kernel));
        % Check sum
        disp(['Kernel sum: ',num2str(sum(sum(kernel)))]);
        
        % Convolve
        out_array = conv2(in_array,kernel,'same');
    case 3
        disp('Input array: 3 dimensions');
        % Create a Gaussian kernel
        kernel = zeros(size_of_kernel,size_of_kernel,size_of_kernel);
        if mod(size_of_kernel,2) == 0 %case where kernel size is even
            center_of_kernel = size_of_kernel/2 + 0.5;
        else %case where kernel size is odd
            center_of_kernel = ceil(size_of_kernel/2);
        end

        for i = 1:size_of_kernel
            for j = 1:size_of_kernel
                for k = 1:size_of_kernel
                    kernel(j,i,k) = exp(-.5 * (((i-center_of_kernel)/sigma_x)^2 + ((j-center_of_kernel)/sigma_y)^2 + ((k-center_of_kernel)/sigma_z)^2));
                end
            end
        end
        
        % Normalize kernel
        kernel = kernel/sum(sum(sum(kernel)));
        % Check sum
        disp(['Kernel sum: ',num2str(sum(sum(sum(kernel))))]);
        
        % Convolve
        out_array = convn(in_array,kernel,'same');
    otherwise
        disp('Not a 2D or 3D array');
end

