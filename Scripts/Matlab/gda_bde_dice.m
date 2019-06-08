%A MATLAB Toolbox
%
%Compare two segmentation results using
%1. Global Consistency Error
%2. Boundary Displacement Error
%
%IMPORTANT: The two input images must have the same size!
%
%Authors: John Wright, and Allen Y. Yang
%Contact: Allen Y. Yang <yang@eecs.berkeley.edu>
%
%(c) Copyright. University of California, Berkeley. 2007.
%
%Notice: The packages should NOT be used for any commercial purposes
%without direct consent of their author(s). The authors are not responsible
%for any potential property loss or damage caused directly or indirectly by the usage of the software.

pkg load image
clear


%Functions required for the calculations
function gce = GlobalConsistencyError(sampleLabels1,sampleLabels2)
  [imWidth,imHeight]=size(sampleLabels1);
  [imWidth2,imHeight2]=size(sampleLabels2);
  N=imWidth*imHeight;
  if (imWidth~=imWidth2)||(imHeight~=imHeight2)
    disp( 'Input sizes: ' );
    disp( size(sampleLabels1) );
    disp( size(sampleLabels2) );
    error('Input sizes do not match in compare_segmentations.m');
  end

  % make the group indices start at 1
  if min(min(sampleLabels1)) < 1
    sampleLabels1 = sampleLabels1 - min(min(sampleLabels1)) + 1;
  end
  if min(min(sampleLabels2)) < 1
    sampleLabels2 = sampleLabels2 - min(min(sampleLabels2)) + 1;
  end

  segmentcount1=max(max(sampleLabels1));
  segmentcount2=max(max(sampleLabels2));

  % compute the count matrix from this we can quickly compute rand index, GCE, VOI, ect...
  n=zeros(segmentcount1,segmentcount2);

  for i=1:imWidth
    for j=1:imHeight
      u=sampleLabels1(i,j);
      v=sampleLabels2(i,j);
      n(u,v)=n(u,v)+1;
    end
  end

  gce = global_consistancy_error(n);
end

% global consistancy error (from BSDS ICCV 01 paper) ... lower => better
function gce = global_consistancy_error(n)
  N = sum(sum(n));
  marginal_1 = sum(n,2);
  marginal_2 = sum(n,1);
  % the hackery is to protect against cases where some of the marginals are
  % zero (should never happen, but seems to...)
  E1 = 1 - sum( sum(n.*n,2) ./ (marginal_1 + (marginal_1 == 0)) ) / N;
  E2 = 1 - sum( sum(n.*n,1) ./ (marginal_2 + (marginal_2 == 0)) ) / N;
  gce = min( E1, E2 );
end

function loss = dice_multi(Y, T)
  % loss = dice_multi(Y, T) returns the Dice loss between
  % the predictions Y and the training targets T. 
  weights = [0.4, 0.4, 0.2];
  
  [argvalue, argmax] = max(Y,[],3);
  
  new_img = zeros(size(Y));
  
  for i = 1:size(Y,1)
    for j = 1:size(Y,2)
      new_img(i,j,argmax(i,j)) = 255;
    end
  end
  
  new_img = new_img/255;
  T = T/255;
  
  dice_class_1 = dice(new_img(:,:,1), T(:,:,1));
  dice_class_2 = dice(new_img(:,:,2), T(:,:,2));
  dice_class_3 = dice(new_img(:,:,3), T(:,:,3));

  loss = dice_class_1 * weights(1) + dice_class_2 * weights(2) + dice_class_3 * weights(3);
end

function z = dice(Y,T)
  z = 2*nnz(Y&T)/((nnz(Y) + nnz(T))+0.000001);
  if (isnan(z))
    z = 1;
  end
end

function averageError = compare_image_boundary_error(imageLabels1, imageLabels2)
  [imageX, imageY] = size(imageLabels1);
  if imageX~=size(imageLabels2,1) | imageY~=size(imageLabels2,2)
    error('The sizes of the two comparing images must be the same.');
  end

  if isempty(find(imageLabels1~=imageLabels1(1)))
    % imageLabels1 only has one group
    boundary1 = zeros(size(imageLabels1));
    boundary1(1,:) = 1;
    boundary1(:,1) = 1;
    boundary1(end,:) = 1;
    boundary1(:,end) = 1;
  else
    % Generate boundary maps
    [cx,cy] = gradient(imageLabels1);
    [boundaryPixelX{1},boundaryPixelY{1}] = find((abs(cx)+abs(cy))~=0);

    boundary1 = abs(cx) + abs(cy) > 0;
  end

  if isempty(find(imageLabels2~=imageLabels2(1)))
    % imageLabels2 only has one group
    boundary2 = zeros(size(imageLabels2));
    boundary2(1,:) = 1;
    boundary2(:,1) = 1;
    boundary2(end,:) = 1;
    boundary2(:,end) = 1;    
  else    
    % Generate boundary maps
    [cx,cy] = gradient(imageLabels2);
    [boundaryPixelX{2},boundaryPixelY{2}] = find((abs(cx)+abs(cy))~=0);

    boundary2 = abs(cx) + abs(cy) > 0;
  end

  % boundary1 and boundary2 are now binary boundary masks. compute their
  % distance transforms:
  D1 = bwdist(boundary1);
  D2 = bwdist(boundary2);

  % compute the distance of the pixels in boundary1 to the nearest pixel in
  % boundary2:
  dist_12 = sum(sum(boundary1 .* D2 ));
  dist_21 = sum(sum(boundary2 .* D1 ));

  avgError_12 = dist_12 / sum(sum(boundary1));
  avgError_21 = dist_21 / sum(sum(boundary2));

  averageError = (avgError_12 + avgError_21) / 2;
end



% enable - disable for the desire calculation
compute_ABDE = true;
compute_AGCE = true;
compute_ADice = true;

% Defines k of kfolds applied and the size of each partition %
kfolds = 1;
partition_size = idivide(35, kfolds, 'floor');
partition_begin = 1;
partition_end = partition_size;

for k=0:(kfolds-1)
  %Input directory path
  path_input_a = 'C:\Users\ErickOF\Documents\Pruebas\predictions\ca_ausm2\border\';
  imagefile_input = dir(strcat(path_input_a,'*.png'));
      
  %Ground Truth directory path
  path_gt_a = 'C:\Users\ErickOF\Documents\Pruebas\gt\border\';
  imagefile_gt = dir(strcat(path_gt_a,'*.png'));
      
  if length(imagefile_input) == 0
    compute_ADice = false;
    Dice(k+1) = 0;
  else
    compute_ADice = true;
  end

  %Average Border Displacement Error
  ABDE = 0;

  %Average Global Consistency Error
  AGCE = 0;
    
  %Average Multi-Class Weighted Dice Loss
  ADice = 0;
    
  out = 0;
    
  if(compute_ABDE)
    for i=partition_begin:partition_end
      % Obtains the i-th id and the image from it
      path_input = imagefile_input(i);
      path_gt = imagefile_gt(i);
      input = double(imread(strcat(path_input_a, '\', path_input.name)));
      gt = double(imread(strcat(path_gt_a,'\', path_gt.name)));

      result_red = compare_image_boundary_error(input(:,:,1),gt(:,:,1));
      result_green = compare_image_boundary_error(input(:,:,2),gt(:,:,2));
      result_blue = compare_image_boundary_error(input(:,:,3),gt(:,:,3));

      BDE = (result_red * 0.3) + (result_green * 0.3) + (result_blue * 0.4);

      ABDE = ABDE + BDE;
    end

    BDE(k+1) = ABDE / partition_size;
  end

  if(compute_AGCE)
    for i=partition_begin:partition_end
      % Obtains the i-th id and the image from it
      path_input = imagefile_input(i);
      path_gt = imagefile_gt(i);
      input = double(imread(strcat(path_input_a, '\', path_input.name)));
      gt = double(imread(strcat(path_gt_a,'\', path_gt.name)));

      result_red = GlobalConsistencyError(input(:,:,1),gt(:,:,1));
      result_green = GlobalConsistencyError(input(:,:,2),gt(:,:,2));
      result_blue = GlobalConsistencyError(input(:,:,3),gt(:,:,3));

      GCE = (result_red * 1/3) + (result_green * 1/3) + (result_blue * 1/3);

      AGCE = AGCE + GCE;
    end

    GCE(k+1) = AGCE / partition_size;
  end
    
  if(compute_ADice)
    nan_results = 0;

    for i=partition_begin:partition_end
      % Obtains the i-th id and the image from it
      path_input = imagefile_input(i);
      path_gt = imagefile_gt(i);
      input = double(imread(strcat(path_input_a, '\', path_input.name)));
      gt = double(imread(strcat(path_gt_a,'\', path_gt.name)));

      Local_Dice = dice_multi(input, gt);

      if(isnan(Local_Dice))
        nan_results = nan_results + 1;
      else
        ADice = ADice + Local_Dice;
      end
    end

    Dice(k+1) = ADice / (partition_size-nan_results);
  end
  
  partition_begin = partition_end;
  partition_end = partition_end + partition_size;
endfor

BDE
GCE
Dice
