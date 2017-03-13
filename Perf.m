%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% (C) Eamonn Keogh, CHL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script can test the 1nn performance of given screen set
clear;
TRAIN = load('CS235testdata4.txt'); % Only one lines need to be changed to test a different dataset  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TRAIN_class_labels = TRAIN(:,1);     % Pull out the class labels.
TRAIN(:,1) = [];                     % Remove class labels from training set.

screen = [55,87,41];
TRAIN = TRAIN(:,screen);
zTRAIN = zscore(TRAIN);
LeaveOne(TRAIN, TRAIN_class_labels)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here is a sample classification algorithm, it is the simple (yet very competitive) one-nearest
% neighbor using the Euclidean distance.
% If you are advocating a new distance measure you just need to change the line marked "Euclidean distance"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels,unknown_object, excludeIndex)
best_so_far = inf;
 
 for i = 1 : length(TRAIN_class_labels)
     if (i ~= excludeIndex)
        compare_to_this_object = TRAIN(i,:);
        distance = sqrt(sum((compare_to_this_object - unknown_object).^2)); % Euclidean distance
        if distance < best_so_far
          predicted_class = TRAIN_class_labels(i);
         best_so_far = distance;
        end
     end
 end;
end

function ans = LeaveOne(TRAIN, TRAIN_class_labels)
    correct = 0;
    for i = 1 : length(TRAIN_class_labels) % Loop over every instance in the test set
       classify_this_object = TRAIN(i,:);
       this_objects_actual_class = TRAIN_class_labels(i);
       predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels, classify_this_object, i);
       if predicted_class == this_objects_actual_class
           correct = correct + 1;
       end;
       if (mod(i,1000)==0)
        disp([int2str(i), ' out of ', int2str(length(TRAIN_class_labels)), ' done']); % Report progress
       end;
    end;
    ans = correct;
end


function ans = HoldOut(TRAIN, TRAIN_class_labels, TEST, TEST_class_labels)
    correct = 0;
    for i = 1 : length(TEST_class_labels) % Loop over every instance in the test set
       classify_this_object = TEST(i,:);
       this_objects_actual_class = TEST_class_labels(i);
       predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels, classify_this_object, 0);
       if predicted_class == this_objects_actual_class
           correct = correct + 1;
       end;
       if (mod(i,1000)==0)
        disp([int2str(i), ' out of ', int2str(length(TEST_class_labels)), ' done']); % Report progress
       end;
    end;
    ans = correct;
end