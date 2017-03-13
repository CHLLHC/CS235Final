%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% (C) Eamonn Keogh, CHL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
filename = 'CS235testdata4.txt' % Only one lines need to be changed to test a different dataset  %
TRAIN = load(filename); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TRAIN_class_labels = TRAIN(:,1);     % Pull out the class labels.
TRAIN(:,1) = [];                     % Remove class labels from training set.

zTRAIN = zscore(TRAIN);
data = [TRAIN_class_labels zTRAIN];

current_set_of_features = []; % Initialize an empty set
curr_acc = [];
best_of_all = [];
best_of_all_acc = 0;


for i = 1 : size(data,2)-1 
    disp(['On the ',num2str(i),'th level of the search tree'])
    feature_to_add_at_this_level = [];
    best_so_far_accuracy    = 0;    
    
     for k = 1 : size(data,2)-1 
       if isempty(intersect(current_set_of_features,k)) % Only consider adding, if not already added.
        %disp(['--Considering adding the ', num2str(k),' feature']);
        accuracy = leave_one_out_cross_validation(data,current_set_of_features,k+1);
        %disp(['---Got ',num2str(accuracy)]);
        if accuracy > best_so_far_accuracy 
            best_so_far_accuracy = accuracy;
            feature_to_add_at_this_level = k;            
        end        
       end
     end
    
    current_set_of_features(i) =  feature_to_add_at_this_level;
    curr_acc(i) = best_so_far_accuracy;
     
     if best_so_far_accuracy > best_of_all_acc
        best_of_all_acc = best_so_far_accuracy;
        best_of_all = current_set_of_features;
     end
    
    disp(['On level ', num2str(i),' i added feature ', num2str(feature_to_add_at_this_level), ' to current set. ACC: ', num2str(best_so_far_accuracy)])
        
end

disp('Best of all set is:');
disp(best_of_all);
disp(['And acc is', num2str(best_of_all_acc)]);

plot(curr_acc)
xlabel('Number of features')
ylabel('Accuracy')
title(['Accuracy Verses Number of Features for ',filename] )
ylim([0 1])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here is a sample classification algorithm, it is the simple (yet very competitive) one-nearest
% neighbor using the Euclidean distance.
% If you are advocating a new distance measure you just need to change the line marked "Euclidean distance"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels,unknown_object, excludeIndex)
best_so_far = inf;
 
 for i = 1 : length(TRAIN_class_labels)
     if (i ~= excludeIndex) % This is for leave-one validation
        compare_to_this_object = TRAIN(i,:);
        distance = sqrt(sum((compare_to_this_object - unknown_object).^2)); % Euclidean distance
        if distance < best_so_far
          predicted_class = TRAIN_class_labels(i);
         best_so_far = distance;
        end
     end
 end;
end

function accuracy = leave_one_out_cross_validation(data,current_set,feature_to_add)
    TRAIN = data;    
    TRAIN_class_labels = TRAIN(:,1);     % Seperate
    TRAIN(:,1) = [];                     
    
    newSet = union(current_set,feature_to_add - 1);
    
    TRAIN = TRAIN(:,newSet);
    
    correct = 0;
    
    for i = 1 : length(TRAIN_class_labels) % Loop over every instance in the test set
       classify_this_object = TRAIN(i,:);
       this_objects_actual_class = TRAIN_class_labels(i);
       predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels,classify_this_object, i);
       if predicted_class == this_objects_actual_class
           correct = correct + 1;
       end;
    end;
    
    accuracy = correct/length(TRAIN_class_labels);
end