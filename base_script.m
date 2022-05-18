input_image_path = "/MATLAB Drive/21_training.tif";
input_mask_path = "/MATLAB Drive/21_mask.jpg";

[segmented_output, superposed_output] = generate_output(input_image_path, input_mask_path);

save_results(segmented_output, superposed_output)



function save_results(segmented_output, superposed_output)
    imshow(segmented_output)
    mkdir('results')
    imwrite(segmented_output,'results/segmentation_output.jpg')
    imwrite(superposed_output,'results/final_superposition_output.jpg')
end





function [step6_output, superposed_output] = generate_output(input_image_path, input_mask_path)
    ff_ = read_image(input_image_path);
    mask_ = read_image(input_mask_path);
    ff_cropped = cropper(ff_);
    mask_cropped = cropper(mask_);
    %imshow(ff_cropped)
    %imshow(mask_cropped)
    step1_output = non_uniform_background_removal(ff_cropped);
    %imshow(step1_output(:,:,2))
    step2_output = pca_color_to_gray(step1_output);
    %imshow(step2_output)
    %step3_output = laplacian_of_gaussian_detector(step2_output);
    step3_output = LoG_edgeDetection(step2_output, 3);
    %imshow(step3_output);
    step4_output = anisodiff2D(step3_output, 2, 0.25, 50, 1);
    max(step4_output, [], 'all')
    %imshow(step4_output)
    seed_point_img = seed_point_generator(step4_output, 5);
    %imshow(seed_point_img)
    step5_output = flood_filled_morphological_reconstruction(seed_point_img, step4_output, mask_cropped(:,:,1));
    %imshow(step5_output)
    step6_output = binarization(step5_output);
    %imshow(step6_output)
    superposed_output = superposition_img_on_mask(step6_output, mask_cropped);
    %imshow(superposed_output);
end






function img = read_image(file_path)
   img = imread(file_path); 
end





function step3_output = laplacian_of_gaussian_detector(img)
    step3_output = edge(img,'log');
end






function step2_output = pca_color_to_gray(img)
    step2_output = img(:,:,2);
end






function step1_output = non_uniform_background_removal(img)
    [R_input, G_input, B_input] = imsplit(img);
    structuring_element = strel('disk',11);
    R_output = step1_helper_function(R_input, structuring_element);
    G_output = step1_helper_function(G_input, structuring_element);
    B_output = step1_helper_function(B_input, structuring_element);
    step1_output = cat(3, R_output, G_output, B_output);
    end






function background_removed_channel_output = step1_helper_function(single_channel_img, structuring_element)
    close_img = imclose(single_channel_img, structuring_element);
    mean_closing_img = mean(close_img, 'all');
    background_removed_channel_output = single_channel_img - (close_img - mean_closing_img);
end






function cropped_image = cropper(img)
    [r, c, nChannels] = size(img);
    sq_dimension = min(r,c);
    targetSize = [sq_dimension, sq_dimension];
    crop_object = centerCropWindow2d(size(img),targetSize);
    cropped_image = imcrop(img, crop_object);
end







function edges = LoG_edgeDetection(Image, sigma)
    % This is a simple implementation of the LoG edge detector. 
    % By: Hossein ZIAEI NAFCHI
    % Image: Gray-level input image
    % sigma: try values like 1, 2, 4, 8, etc.
    % edges: Output edge map
    % Form the LoG filter
    nLoG = filter_LoG(sigma);
    convResult = imfilter(double(Image), nLoG, 'replicate');
    slope = 0.5 * mean( abs(convResult(:)) );
    %% Vertical edges
    % Shift image one pixel to the left and right
    convLeft = circshift(convResult, [0, -1]);
    convRight = circshift(convResult, [0, 1]);
    % The vertical edges (-  +, +  -, - 0 +, + 0 -)
    v_edge1 = convResult < 0 & convLeft > 0 & abs(convLeft - convResult) > slope;
    v_edge2 = convResult < 0 & convRight > 0 & abs(convRight - convResult) > slope;
    v_edge3 = convResult == 0 & sign(convLeft) ~= sign(convRight) & abs(convLeft - convRight) > slope;
    v_edge = v_edge1 | v_edge2 | v_edge3; 
    v_edge(:, [1 end]) = 0;
    %% Horizontal edges
    % Shift image one pixel to the top and bottom
    convTop = circshift(convResult, [-1, 0]);
    convBot = circshift(convResult, [1, 0]);
    % The horizontal edges (-  +, +  -, - 0 +, + 0 -)
    h_edge1 = convResult < 0 & convTop > 0 & abs(convTop - convResult) > slope;
    h_edge2 = convResult < 0 & convBot > 0 & abs(convBot - convResult) > slope;
    h_edge3 = convResult == 0 & sign(convTop) ~= sign(convBot) & abs(convTop - convBot) > slope;
    h_edge = h_edge1 | h_edge2 | h_edge3; 
    h_edge([1 end], :) = 0;
    % Combine vertical and horizontal edges
    edges = v_edge | h_edge;
end








function nLoG = filter_LoG(sigma)
    % This function generates Laplacian of Gaussian filter given the sigma
    % parameter as its input. Filter size is estimated by multiplying the 
    % sigma with a constant.
    % By: Hossein ZIAEI NAFCHI
    %% The function
    fsize = ceil(sigma) * 5; % filter size is estimated by: sigma * constant
    fsize = (fsize - 1) / 2; 
    [x, y] = meshgrid(-fsize : fsize, -fsize : fsize);
    % The two parts of the LoG equation
    a = (x .^ 2 + y .^ 2 - 2 * sigma ^ 2) / sigma ^ 4;
    b = exp( - (x .^ 2 + y .^ 2) / (2 * sigma ^ 2) );
    b = b / sum(b(:));
    % The LoG filter
    LoG = a .* b;
    % The normalized LoG filter
    nLoG = LoG - mean2(LoG);
    % ** end of the function ** 
    %% Uncomment to display LoG plot 
    % surf(x, y , nLoG);
    % xlabel('x'); 
    % ylabel('y');
    % zlabel('LoG'); 
    % name = ['Laplacian of 2D Gaussian ( \sigma = ' , num2str(sigma), ' )']; 
    % title(name);
    %% Uncomment to save at output
    % print(num2str(sigma), '-dpng', '-r400');
end








function diff_im = anisodiff2D(im, num_iter, delta_t, kappa, option)
    %ANISODIFF2D Conventional anisotropic diffusion
    %   DIFF_IM = ANISODIFF2D(IM, NUM_ITER, DELTA_T, KAPPA, OPTION) perfoms 
    %   conventional anisotropic diffusion (Perona & Malik) upon a gray scale
    %   image. A 2D network structure of 8 neighboring nodes is considered for 
    %   diffusion conduction.
    % 
    %       ARGUMENT DESCRIPTION:
    %               IM       - gray scale image (MxN).
    %               NUM_ITER - number of iterations. 
    %               DELTA_T  - integration constant (0 <= delta_t <= 1/7).
    %                          Usually, due to numerical stability this 
    %                          parameter is set to its maximum value.
    %               KAPPA    - gradient modulus threshold that controls the conduction.
    %               OPTION   - conduction coefficient functions proposed by Perona & Malik:
    %                          1 - c(x,y,t) = exp(-(nablaI/kappa).^2),
    %                              privileges high-contrast edges over low-contrast ones. 
    %                          2 - c(x,y,t) = 1./(1 + (nablaI/kappa).^2),
    %                              privileges wide regions over smaller ones. 
    % 
    %       OUTPUT DESCRIPTION:
    %                DIFF_IM - (diffused) image with the largest scale-space parameter.
    % 
    %   Example
    %   -------------
    %   s = phantom(512) + randn(512);
    %   num_iter = 15;
    %   delta_t = 1/7;
    %   kappa = 30;
    %   option = 2;
    %   ad = anisodiff2D(s,num_iter,delta_t,kappa,option);
    %   figure, subplot 121, imshow(s,[]), subplot 122, imshow(ad,[])
    % 
    % See also anisodiff1D, anisodiff3D.
    % References: 
    %   P. Perona and J. Malik. 
    %   Scale-Space and Edge Detection Using Anisotropic Diffusion.
    %   IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    %   12(7):629-639, July 1990.
    % 
    %   G. Grieg, O. Kubler, R. Kikinis, and F. A. Jolesz.
    %   Nonlinear Anisotropic Filtering of MRI Data.
    %   IEEE Transactions on Medical Imaging,
    %   11(2):221-232, June 1992.
    % 
    %   MATLAB implementation based on Peter Kovesi's anisodiff(.):
    %   P. D. Kovesi. MATLAB and Octave Functions for Computer Vision and Image Processing.
    %   School of Computer Science & Software Engineering,
    %   The University of Western Australia. Available from:
    %   <http://www.csse.uwa.edu.au/~pk/research/matlabfns/>.
    % 
    % Credits:
    % Daniel Simoes Lopes
    % ICIST
    % Instituto Superior Tecnico - Universidade Tecnica de Lisboa
    % danlopes (at) civil ist utl pt
    % http://www.civil.ist.utl.pt/~danlopes
    %
    % May 2007 original version.
    % Convert input image to double.
    im = double(im);
    % PDE (partial differential equation) initial condition.
    diff_im = im;
    % Center pixel distances.
    dx = 1;
    dy = 1;
    dd = sqrt(2);
    % 2D convolution masks - finite differences.
    hN = [0 1 0; 0 -1 0; 0 0 0];
    hS = [0 0 0; 0 -1 0; 0 1 0];
    hE = [0 0 0; 0 -1 1; 0 0 0];
    hW = [0 0 0; 1 -1 0; 0 0 0];
    hNE = [0 0 1; 0 -1 0; 0 0 0];
    hSE = [0 0 0; 0 -1 0; 0 0 1];
    hSW = [0 0 0; 0 -1 0; 1 0 0];
    hNW = [1 0 0; 0 -1 0; 0 0 0];
    % Anisotropic diffusion.
    for t = 1:num_iter
            % Finite differences. [imfilter(.,.,'conv') can be replaced by conv2(.,.,'same')]
            nablaN = imfilter(diff_im,hN,'conv');
            nablaS = imfilter(diff_im,hS,'conv');   
            nablaW = imfilter(diff_im,hW,'conv');
            nablaE = imfilter(diff_im,hE,'conv');   
            nablaNE = imfilter(diff_im,hNE,'conv');
            nablaSE = imfilter(diff_im,hSE,'conv');   
            nablaSW = imfilter(diff_im,hSW,'conv');
            nablaNW = imfilter(diff_im,hNW,'conv'); 
            
            % Diffusion function.
            if option == 1
                cN = exp(-(nablaN/kappa).^2);
                cS = exp(-(nablaS/kappa).^2);
                cW = exp(-(nablaW/kappa).^2);
                cE = exp(-(nablaE/kappa).^2);
                cNE = exp(-(nablaNE/kappa).^2);
                cSE = exp(-(nablaSE/kappa).^2);
                cSW = exp(-(nablaSW/kappa).^2);
                cNW = exp(-(nablaNW/kappa).^2);
            elseif option == 2
                cN = 1./(1 + (nablaN/kappa).^2);
                cS = 1./(1 + (nablaS/kappa).^2);
                cW = 1./(1 + (nablaW/kappa).^2);
                cE = 1./(1 + (nablaE/kappa).^2);
                cNE = 1./(1 + (nablaNE/kappa).^2);
                cSE = 1./(1 + (nablaSE/kappa).^2);
                cSW = 1./(1 + (nablaSW/kappa).^2);
                cNW = 1./(1 + (nablaNW/kappa).^2);
            end
    end
            % Discrete PDE solution.
            diff_im = diff_im + ...
                      delta_t*(...
                      (1/(dy^2))*cN.*nablaN + (1/(dy^2))*cS.*nablaS + ...
                      (1/(dx^2))*cW.*nablaW + (1/(dx^2))*cE.*nablaE + ...
                      (1/(dd^2))*cNE.*nablaNE + (1/(dd^2))*cSE.*nablaSE + ...
                      (1/(dd^2))*cSW.*nablaSW + (1/(dd^2))*cNW.*nablaNW );
               
            % Iteration warning.
            %fprintf('\rIteration %d\n',t);
end







function seed_points = seed_point_generator(img, box_size)
    [r, c] = size(img);
    seed_points = zeros(r, c);
    for i=1:box_size:r
        for j=1:box_size:c
            x = zeros(25,2);
            cnt=0;
            for i_ = i:min(i+box_size, r)
                for j_ = j:min(j+box_size, c)
                    isgreater = img(i_, j_) > 0.2;
                    if isgreater==true
                        cnt = cnt+1;
                        x(cnt,:) = [i_,j_];
                    end
                end
            end

            if cnt==0
                continue
            end

            idx = mod(uint8(i+j),uint8(cnt))+1;
            seed_points(x(idx,1), x(idx,2)) = 1.0;
        end
    end
end







function step5_output = flood_filled_morphological_reconstruction(marker, conne, mask)
    mask = im2double(mask);
    step5_output = imreconstruct(marker,mask);
end







function final_output = binarization(img)
    final_output = imbinarize(img);
end







function superposed_image = superposition_img_on_mask(img, mask)
    superposed_image = cat(3, uint8(img)*255, uint8(img)*255, uint8(img)*255);
    superposed_image(:,:,2) = mask;
end
