% Matthew Conrad
% 11/27/2017
%% Look at final_degraded_image.nrrd for idea of scale
clear all; clc; close all; imtool close all;
image = nrrdread('final_degraded_image.nrrd');
image_max = max(reshape(image,[1 256*256]));
image_min = min(reshape(image,[1 256*256]));
imtool(image,[]);

% To understand the system, we need to understand the intensity scale it
% operates on. I was doing values between 0 and 1 and ran into issues. The
% max for this image is 692.3309. The following delta functions will be
% scaled so that the impulse has an amplitude of 692.3309.  
%% Test system for SI or SV with delta functions
clear all; clc; close all; imtool close all;

% Create an array with 49 delta functions for input
input_image = createPointSourceGrid(1,256,7);

% Display input
imtool(input_image,[]);

% Calculate output of each system for the input_image
output_image = system4(input_image);

% Display outputs
imtool(output_image,[]);

% Test gives evidence for Shift Variance because the delta functions in the
% corners of the grid were the least attenuated; whereas, the ones in the
% middle were more attenuated. Although the ones in the middle were more
% attenuated, they were still distinguishable, but still closer to the
% noise floor than the deltas in the corners. This may mean the Shift
% Varaince is radial. The next test will potentially provide more insight.

%% Test system for SI or SV with squares instead of delta functions
clear all; clc; close all; imtool close all;

% Create a point source grid image 
input_image = createPointSourceGrid(15,256,5);

% Display input
imtool(input_image,[]);

% Calculate output of each system for the input_image
output_image = system4(input_image);

% Display outputs
imtool(output_image,[]);

% Test was useful because it showed that the corner squares were seemingly
% smeared away from the middle whereas the square in the middle was just
% blurred. The square in the corner seemed as if they were blurred and
% shifted, it is hard to tell. However, this proves that the system is
% Shift Variant. Seems to be radial SV, but not conclusive. It also seems
% the smearing is only along the y=x and y=-x lines and not along the x and
% y axes. This needs further investigation. 

%% Test for linearity
clear all; clc; close all; imtool close all;

% Create input images 
input1_image = createPointSourceGrid(15,256,3); % This represents x1(t)
input2_image = createPointSourceGrid(9,256,5); % This represents x2(t)
imtool([input1_image input2_image],[]); % Display both

% Combine scaled versions of image 1 & 2 to get a 3rd input image
% This represents x3(t) = alpha*x1(t)+beta*x2(t);
input3_image = .5*input1_image + .5*input2_image; 
imtool(input3_image,[]);

% Calculate output of the system 
output1_image = system4(input1_image); % These represent y1(t)
output2_image = system4(input2_image); % These represent y2(t)
imtool([output1_image output2_image],[]); % Display both

% Calculate output of each system for input image 3
% These represent H{alpha*x1(t)+beta*x2(t)} in the linearity formula
output3_image = system4(input3_image);

% Combine the outputs of images 1 & 2 
% This represents alpha*y1(t)+beta*y2(t)
output4_image = .5*output1_image + .5*output2_image;

% Compare output3's to output4's to check linearity
linear = isequal(output3_image,output4_image);

% Display the images side by side
imtool(output3_image,[]); imtool(output4_image,[]);

% In this analysis, it shows system4 is nonlinear, but this result in
% unreliable since isequal checks for exact accuracy, however error could
% prevent this. Looking at output3_image and output4_image, it appears they
% are exactly the same meaning the system4 is linear. A more quantitative
% measurement could provide further insight. 

% Find the difference between output images
difference = output3_image - output4_image;

% Take the sum of the difference array to get the total error 
sum = sum(sum(difference));

% The total error in the two output images described by the linearity
% principle was found to be 21146. In Programming Assignment 3, we found
% that the highest error was 31 and that system was revealed to be linear.
% Keep in mind that the 31 was on a different intensity scale of 0 to 1 for
% a 512x512 image compared to 0 to 692.3309 for a 256x256 image. Dividing
% the total errors by the total possible intensity sum (calculated width x
% height x intensity_max) the relative errors came to be on the order of
% 10^-4, so the two errors are quantitatively similar and visually the
% same, so we can conclusively determine that system4 is linear.

%% Could find FWHM, but since system is SV, we'd have to do so for many positions,
% To be tackled later once the noise is characterized since FWHM works best
% on a noiseless signal.

%% Experiment into how the FFT is affected using gaussian image
clear all; clc; close all; imtool close all;

% Create a point source
point_source = createPointSourceGrid(1,256,1);

% Apply gaussian blur
input_image = gaussian_blur(point_source,3,3,0,25);

% Run input image through system
output_image = system4(input_image);

% Display the two
imtool(input_image,[]);
imtool(output_image,[]);

% Take FFTs
input_image_fft = fft2(input_image);
output_image_fft = fft2(output_image);

% Display the ffts together
imtool(fftshift(abs(input_image_fft)), []);
imtool(fftshift(abs(output_image_fft)), []);

% The input and output images aren't very different from what we've seen in
% the past. As expected, the FFT of the input is another Gaussian. However,
% the FFT of the output after system4 appears as noise. It was not until
% after seeing this result that I remembered FFT cannot be directly applied 
% for a shift-variant problem. This was confirmed after seeing that fact in
% a PhD thesis.

%% Test system4 with a non-point-source image
clear all; clc; close all; imtool close all;

% Import image
in_image = imread('Einstein.jpg');
out_image = system4(in_image);

% Display images
imtool(in_image,[]);
imtool(out_image,[]);

% This shows that a normal image gets deteriorated with noise and blurring
% at the least. Future tests should be done to see how the FFT and similar
% characteristics are changed by the system for typical images. 

%% Test for stationarity and ergodicity
clear all; clc; close all; imtool close all;

% Create an image of constant intensity so that any variation in the output
% image is due to the random process. If we just gave the process an image
% like the Einstein one, then impressions of that image would exist in the
% output image and are not characteristic of the system.
in_image = ones(256,256);
in_image = in_image.*255;

% Create variables 
number_of_samples = 10;

% Allocate space for the samples
samples = zeros(256,256,number_of_samples);

% Take number_of_samples samples for the purpose of estimating mean image
for i = 1:number_of_samples
    samples(:,:,i) = system4(in_image);
end

% Mean image estimate
mean_image = mean(samples,3);

total_variance_image = zeros(256,256);
for i = 1:number_of_samples
    total_variance_image = total_variance_image + (samples(:,:,i)-mean_image).^2;
end
variance_image = total_variance_image./number_of_samples;
% Display images
imtool(mean_image,[]); imtool(variance_image,[]);

% Calculate measurements to determine ergodicity
f_bar = mean(reshape(samples(:,:,i),[256*256,1]));
mean_image_average = mean(reshape(mean_image,[256*256,1]));

% The mean and variance images appears to maintain a constant structure
% throughout the ensemble statistic images. Thus, by visual inspection, I
% conclude this system is stationary. Also, the system also appears to be
% ergodic seeing that the f_bar of one sample was approximately equal to
% the constant intensity value of the mean image. 

%% Determine noise distribution
% Calculate spatial mean and variance, and a histogram
% THIS SECTION MUST RUN AFTER THE SECTION ABOVE ON STATIONARITY

out_image = samples(:,:,1);

% Since the input is a constant image, many operations (such as
% blur, derivative, etc.) have no affect on the input meaning
% the output = input + noise. There are still several things
% (such as introducing notch frequencies) that can cause
% this to be false, however none have been seen in any experiments thus
% far. By assuming we can ignore the deterministic aspects of the system,
% we can characterize the noise accurately.

noise_image = out_image - in_image; % Essentially the noise image
imtool(in_image,[]); imtool(out_image,[]); 

% Plot the histogram of the noise image
hist(reshape(noise_image,[256*256,1]),100);

% Calculate the spatial mean and variance of the noise image
spatial_mean = mean(reshape(noise_image,[256*256,1]));
spatial_var = var(reshape(noise_image,[256*256,1]));

% Looking at the histogram of the noise image, we see that the noise takes
% on a Poisson distribution with a lower end tail, spatial mean of -4, and
% spatial variance of 392.0484.  

%% Calculate autocorrelation and PSD of the noise image

% THIS SECTION MUST RUN AFTER THE SECTION ABOVE ON NOISE DISTRIBUTION

% Calculate the autocorrelation of the noise image
noise_image_Rnn = xcorr2(noise_image);

% Display autocorrelation images
imtool(noise_image_Rnn,[]);

% Plot a 1D "slice" through the autocorrelation images along the x-axis
figure; plot(-255:255,noise_image_Rnn(255,:));
title('1D Slice of Autocorrelation of the Noise along x-axis');
xlabel('Pixel Difference (in pixels)'); 

% Although we cannot use the FFT for the entire system, we can use it for
% the noise since that is stationary. We can take the FFT of Rnn to get PSD. 
noise_image_PSD = fft2(noise_image_Rnn);
imtool(fftshift(abs(noise_image_PSD)),[]);

% Looking at the autocorrelation image and the 1D plot, it appears that the
% noise is correlated mainly along the x and y axes, and not much else
% anywhere. The PSD shows that the signal content is also along the x
% and y axes. 

% Unfortunately, we cannot conclude this as white noise since
% Rff is not a delta function. 

%%
clear all; clc; close all; imtool close all;

% Read in the image
degraded_image = rgb2gray(imread('degraded_image.jpg'));

% Take the FFT
degraded_image_fft = fftshift(fft2(degraded_image));
imtool(abs(degraded_image_fft));

% Create a custom binary filter, then smooth it with Gaussian blur
filt = ones(256,256);
zero_zone = zeros(10,90);
filt(1:90,128-4:128+5) = zero_zone';
filt(256-89:256,128-4:128+5) = zero_zone';
filt(128-4:128+5,1:90) = zero_zone;
filt(128-4:128+5,256-89:256) = zero_zone;
filt = gaussian_blur(filt,5,5,0,15);

% Filter the degraded image
filtered_image_fft = degraded_image_fft .* filt;
imtool(abs(filtered_image_fft),[]);

filtered_image = ifft2(ifftshift(filtered_image_fft));

% Create variables for the MAP
beta = .1;
mean_image = ones(256,256) * mean(reshape(filtered_image,256^2,1));
iterations = 10;

% Create a point source grid for tiling image and run it through the system
PS_grid = createPointSourceGrid(1,256,8);
PSF_grid = system4(PS_grid);

% Allocate space for the new image
out_image = zeros(256,256);

for i = 1:8
    for j = 1:8
        % Calculate the indices of the full image for the sub-image 
        row_index1 = 256/8*(i-1)+1; row_index2 = 256/8*i;
        col_index1 = 256/8*(j-1)+1; col_index2 = 256/8*j;
        disp(col_index1); disp(col_index2);
        
        % Extract the sub-images 
        sub_image = filtered_image(row_index1:row_index2,col_index1:col_index2);
        PSF_sub_image = PSF_grid(row_index1:row_index2,col_index1:col_index2);
        sub_mean_image = mean_image(row_index1:row_index2,col_index1:col_index2);
        
        % Take FFT of the PSF to get the H for the sub-image
        H_sub_image = fft2(PSF_sub_image);
        
        % Reshape for input
        sub_image = reshape(sub_image,32^2,1);
        sub_mean_image = reshape(sub_mean_image,32^2,1);
        H_sub_image = reshape(H_sub_image,32^2,1);
        
        % Make H into a 1024x1024 matrix by making every column the FFT of
        % the PSF for that sub-image
        H_matrix = zeros(1024,1024);
        for k = 1:32^2
            H_matrix(:,k) = H_sub_image;
        end 
               
        [temp_image, residual] = map_norm_iter(sub_image,sub_image,H_matrix,iterations,sub_mean_image,beta);
        
        out_image(row_index1:row_index2,col_index1:col_index2) = reshape(temp_image,32,32);
    end
end

imtool(abs(out_image),[]);