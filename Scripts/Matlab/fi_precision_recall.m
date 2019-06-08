pkg load image;
clear;


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
      
  A_precision = 0; A_recall = 0; A_f1 = 0; 
        
  for i=partition_begin:partition_end
    % Obtains the i-th id and the image from it
    path_input = imagefile_input(i);
    path_gt = imagefile_gt(i);
    input = double(imread(strcat(path_input_a, '\', path_input.name)));
    gt = double(imread(strcat(path_gt_a,'\', path_gt.name)));
        
    # 1 if result is closer to 255, 0 otherwise
    input = (input > 0);
    gt = (gt > 0);
        
    TP_Matrix = input & gt;
    TP = sum(sum(TP_Matrix));
        
    FP_Matrix = (gt - input) > 0;
    FP = sum(sum(FP_Matrix));
       
    FN_Matrix = (input - gt) > 0;
    FN = sum(sum(FN_Matrix)); 

    epsilon = 0.0000001;
        
    %precision = (TP+epsilon) ./ ((TP+FP) + epsilon); 
    %recall = (TP+epsilon) ./ ((TP+FN) + epsilon);
    precision = ((TP+epsilon) ./ ((TP+FP) + epsilon))*[0.3; 0.3; 0.4]; 
    recall = ((TP+epsilon) ./ ((TP+FN) + epsilon))*[0.3; 0.3; 0.4];
        
    f1 = 2*((precision .* recall) ./ (precision + recall));
        
    A_precision = A_precision + precision; 
    A_recall = A_recall + recall; 
    A_f1 = A_f1 + f1; 
  endfor
        
  A_precision = A_precision ./ partition_size
  A_recall = A_recall ./ partition_size
  A_f1 = A_f1 ./ partition_size
    
  %PMat((k+1), :) = reshape(A_precision, 1, 3);
  %PMat((k+1), :) = reshape(A_recall, 1, 3);
  %A_f1((k+1), :) = reshape(A_f1, 1, 3);
      
  partition_begin = partition_end;
  partition_end = partition_end + partition_size;
endfor