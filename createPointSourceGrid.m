function [ out_image ] = createPointSourceGrid( diameter,image_size,ps_grid_size )
%CREATEPOINTSOURCEGRID creates a square image with a grid of point sources
%equally spaced out in the image. Point sources can have varying sizes
%   Params:
%   diameter = The size of the square. Diameter = 3 means square = 3x3 
%   image_size = size of the image. image_size = 256 means image = 256x256
%   ps_grid_size = size of the point source grid. size = 7 means 7x7
%   Note: diameter must be odd
%   Outputs:
%   out_image = the output point source grid


% Create an array with 49 delta functions for input
out_image = zeros(image_size,image_size);  
radius = floor(diameter/2);
for i = 1:ps_grid_size 
    for j = 1:ps_grid_size
        row = ceil(i/(ps_grid_size+1)*image_size);
        col = ceil(j/(ps_grid_size+1)*image_size);
        out_image( (row-radius):(row+radius) , (col-radius):(col+radius)) = 692.3309;
    end
end

end

