clear;
close all;

% Add VLFeat toolbox to MATLAB working path
run('vlfeat-0.9.17-bin/vlfeat-0.9.17/toolbox/vl_setup.m');
addpath(genpath('getNmnistDesc'));

% Put all the folders (classes) in a single folder
train_dataset_path = '../N-SOD Dataset/Train';
test_dataset_path = '../N-SOD Dataset/Test';

filenamesA = dir2(train_dataset_path);

num_classes=numel(filenamesA);
for i=1:num_classes
    classnames{i}=filenamesA(i).name;
end
% create training labels later
train_label=[];

param.descsize = 7; % denotes a N by N region sampled from the subsampled count matrix
param.queuesize = 5000;
param.countmatsubsamp = 2; % a 2 by 2 cell region for the count matrix
param.minNumEvents = 200; % wait for this many events before getting desc from queue

disp(param);

% TD Noise Filter
% Assumes Garrick's AER functions are available (www.garrickorchard.com)
us_time_filter = 5e3; % in micro-seconds
refractory_period = 1e3;

% Debug purpose --
training_done = 0;
training_desc_done = 0; % for debug purposes only , if desc have been stored

% The training phase is here so that it is easy to change parameters
% and check the code of the FPGA implmentation. The testing stage will be
% made modular like how the FPGA implementation is intended to be, fixed
% precision and memory constraints closely followed.

histopts.num_bins = 150; % codebook size
histopts.numSpatialX = [1] ; % ignore but required for code, if [1 2]

% performs spatial pyramid matching (not needed)
histopts.numSpatialY = [1] ; % ignore

% SVM params
svmOpts.C = 10 ;
svmOpts.biasMultiplier = 1 ;

event_chunk_period = 1e4; % get a histogram for classification every 10k events, same for testing

% Check if codebook exists
model_str_stringname = 'modelTD4cl_D5DEMOsplitAUG_FPGAver2nonorm';
try
    load(['./Recognition_FPGA_trainfiles/ECtrainmodels/' model_str_stringname ...
        num2str(histopts.num_bins) num2str(param.countmatsubsamp) num2str(param.descsize)]);
    model_done = 1;
catch
    model_done = 0;
end

if training_done == 0
    count = 0;
    train_desc_savefolder = './Recognition_FPGA_trainfiles/D5DEMOrectFPGA_splitaug/';
    mkdir(train_desc_savefolder);
    if training_desc_done == 0
        for class_i=1:num_classes
            classi_name = cell2mat(classnames(class_i));
            folder_path = fullfile(train_dataset_path,classi_name);
            subfolder_names = dir(folder_path);
            isdir_tags = [subfolder_names.isdir];
            subfolder_names = subfolder_names(~isdir_tags);
            subfolder_names = {subfolder_names.name};
            mkdir(fullfile(train_desc_savefolder, classi_name));
            for train_classi = 1:numel(subfolder_names)
                count = count  + 1;
                filename = subfolder_names{train_classi};
                filepath = fullfile(train_dataset_path,classi_name, filename);
                
                TD = read_linux(filepath);
                TD = ImplementRefraction(TD, refractory_period);
                TD = FilterTD(TD, us_time_filter);
                
                if numel(TD.x) < 3000
                    TD = read_linux(filepath);
                    TD = ImplementRefraction(TD, refractory_period);
                end
                
                trainimage_sizes(count,:) = [max(TD.y) max(TD.x)];
                savepath_file = fullfile(train_desc_savefolder, classi_name, filename(1:end-4));
                
                try
                    load(savepath_file);
                catch
                    [frames, desc] = getDescs_RECT_FPGA(TD, param);
                    
                    savepath_file = fullfile(train_desc_savefolder, classi_name, filename(1:end-4));
                    disp(savepath_file);
                    save(savepath_file,'frames','desc');
                end
                
                train_label = [train_label class_i];
                disp(count);
            end
        end
        
        save('./Recognition_FPGA_trainfiles/D5DEMO4splitAUGdesc_7x7subsamp2x2_ustime5e3.mat',...
            'train_label','trainimage_sizes','-v7.3');
        
    else
        disp('Loading descrs...');
        load('./Recognition_FPGA_trainfiles/D5DEMO4splitAUGdesc_7x7subsamp2x2_ustime5e3.mat');
    end
    
    % Build the codebook
    if model_done == 0
        [train_data, loctrain_label, new_train_label] = readDescs(train_desc_savefolder,20, 'nonorm');
        [model.vocab, model.assoc] = vl_kmeans(vl_colsubset(single([train_data.desc]), 4e6), histopts.num_bins, 'verbose','algorithm', 'ANN') ;
        model.kdtree = vl_kdtreebuild(model.vocab, 'Distance','L1') ;
        model.kdtree = convert_tree_template( model.kdtree );
        model.vocab=double(model.vocab);
        mkdir('./Recognition_FPGA_trainfiles/ECtrainmodels/');
        save(['./Recognition_FPGA_trainfiles/ECtrainmodels/' model_str_stringname ...
            num2str(histopts.num_bins) num2str(param.countmatsubsamp) num2str(param.descsize)],'model','loctrain_label');
    end
    
    % Initialize detection vector
    det_vector = zeros(histopts.num_bins, 1);
    det_class = find(strcmp(classnames, 'Thumper')); % the integer denoting the Thumper class
    
    train_label = [];
    count_sep = 0;
    count = 0;
    
    for class_i=1:num_classes
        classname=classnames{class_i};
        folder_path = fullfile(train_dataset_path,classname);
        subfolder_names = dir(folder_path);
        isdir_tags = [subfolder_names.isdir];
        subfolder_names = subfolder_names(~isdir_tags);
        subfolder_names = {subfolder_names.name};
        for j=1:numel(subfolder_names)
            count = count + 1;
            filename = subfolder_names{j};
            
            filepath=fullfile(train_desc_savefolder, classname, [filename(1:end-4) '.mat']);
            disp(filepath);
            [train_data, loctrain_label, label] = readDescs(filepath, [], 'nonorm');
            start_chunk = 1;
            end_chunk = event_chunk_period;
            for event_c = 1:ceil(size(train_data.frames,2)/event_chunk_period)
                count_sep= count_sep + 1;
                try
                    desc = train_data.desc(:, start_chunk:end_chunk);
                    frames = train_data.frames(:,start_chunk:end_chunk);
                catch
                    desc = train_data.desc(:, start_chunk:end);
                    frames = train_data.frames(:,start_chunk:end);
                end
                hists{count_sep} = get_imagehist_FPGA(model,trainimage_sizes(count,:),...
                    double(frames), double(desc), histopts);
                
                % Evaluate the codewords assigned to the Thumper class
                if class_i == det_class
                    det_vector = det_vector + hists{count_sep};
                elseif class_i == 1  % Background class
                    det_vector = det_vector - hists{count_sep};
                end
                
                train_label = [train_label class_i];
                start_chunk = end_chunk + 1;
                end_chunk = end_chunk + event_chunk_period;
                
                if end_chunk > size(train_data.frames,2)
                    end_chunk = size(train_data.frames,2);
                end
            end
        end
    end
    top_n_codes = 5;
    detector_codewords = find(det_vector > 0 );
    temp_act = det_vector((det_vector > 0 ));
    [~,ind] = sort(temp_act,'descend');
    detector_codewords = detector_codewords(ind(1:top_n_codes));
    
    hists = cat(2, hists{:}) ;
    
    hists=cat(1,hists);
    
    psix = hists; % kernel removed, above line applies kernel
    
    %SVM Calculation
    lambda = 1 / (svmOpts.C * length(train_label)) ;
    w = [] ;
    for ci = 1:length(unique(train_label))
        perm = randperm(length(train_label)) ;
        y = 2 * (train_label == ci) - 1 ;
        [w(:,ci) b(:, ci) info(ci)] = vl_svmtrain(psix(:,perm), ...
            y(perm), lambda, ...
            'MaxNumIterations', 50/lambda, ...
            'BiasMultiplier', svmOpts.biasMultiplier) ;
    end
    
    svmmodel.b = b ;
    svmmodel.w = w;
    save('./Recognition_FPGA_trainfiles/svvmodel_D5DEMOsplitAUG_FPGAver2_150codebokNONORM_newkd_withDet.mat','svmmodel','detector_codewords');
    
else
    % Load the SVM
    load('./Recognition_FPGA_trainfiles/svvmodel_D5DEMOsplitAUG_FPGAver2_150codebokNONORM_newkd_withDet.mat');
    
    % Set precision to 8 significant digits
    digitsOld = digits(8);
    
    svmmodel.w = vpa(svmmodel.w);
    svmmodel.b = vpa(svmmodel.b);
    
    % Load the codebook
    load(['./Recognition_FPGA_trainfiles/ECtrainmodels/' model_str_stringname ...
        num2str(histopts.num_bins) num2str(param.countmatsubsamp) num2str(param.descsize)]);
    
end

%% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% testing stage
count = 0;
count_sep = 0;
test_label = [];

% Detection matrix, similar to the count matrix
det_matrix = zeros(180,240);
temp_count_matrix = det_matrix;
det_class = find(strcmp(classnames, 'Thumper')); % the integer denoting the Thumper class

subsamp_count_matrix = param.countmatsubsamp;
subsamp_desc_regionsize = floor(param.descsize/2); %radius of descsize by descsize region
max_y = 180/subsamp_count_matrix;
max_x = 240/subsamp_count_matrix;

% Read one TD sample at a time and process, can be used to test a sequence of TD files
% for the offline FPGA testing
for class_i=1:num_classes
    classi_name = cell2mat(classnames(class_i));
    folder_path = fullfile(test_dataset_path,classi_name);
    subfolder_names = dir(folder_path);
    isdir_tags = [subfolder_names.isdir];
    subfolder_names = subfolder_names(~isdir_tags);
    subfolder_names = {subfolder_names.name};
    for image_classi = 1:numel(subfolder_names)
        count = count + 1;
        filename = subfolder_names{image_classi};
        filepath = fullfile(test_dataset_path,classi_name, filename);
        
        TD = read_linux(filepath);
        TD.x = TD.x(:);
        TD.y = TD.y(:);
        TD.ts = TD.ts(:);
        TD.p = TD.p(:);
        
        % Saved bin files are not with refraction and FilterTD
        TD = ImplementRefraction(TD, refractory_period);
        TD = FilterTD(TD, us_time_filter);
        
        if numel(TD.x) < 3000
            TD = read_linux(filepath);
            TD = ImplementRefraction(TD, refractory_period);
        end
        
        % initialize count matrix with zero padding at the boundaries
        count_matrix = zeros(max_y + 2* subsamp_desc_regionsize, max_x + 2*subsamp_desc_regionsize);
        
        global_eventcount = 0;
        desc_count = 0;
        
        queue = CQueue();
        queue.capacity = param.queuesize;
        minNumEvents = param.minNumEvents;
        
        bow_histogram = zeros(histopts.num_bins, 1);
        distances_sqd =  zeros(1, histopts.num_bins);
        
        % Simulate the process of event-by-event processing
        for i = 1:size(TD.x,1)
            
            global_eventcount = global_eventcount + 1;
            
            % Account for subsampled counting and zero padding
            event_x = ceil(TD.x(i)./subsamp_count_matrix) + subsamp_desc_regionsize;
            event_y = ceil(TD.y(i)./subsamp_count_matrix) + subsamp_desc_regionsize;
            
            if global_eventcount <= minNumEvents
                queue.push([event_x event_y]);
                count_matrix(event_y, event_x) = count_matrix(event_y, event_x) + 1;
            else
                if global_eventcount > param.queuesize
                    old_xy = queue.pop();
                    count_matrix(old_xy(2), old_xy(1)) = count_matrix(old_xy(2), old_xy(1)) - 1;
                end
                queue.push([event_x event_y]);
                count_matrix(event_y, event_x) = count_matrix(event_y, event_x) + 1;
                
                desc = count_matrix(event_y-subsamp_desc_regionsize:event_y + subsamp_desc_regionsize, ...
                    event_x - subsamp_desc_regionsize:event_x + subsamp_desc_regionsize);
                desc = desc(:);
                
                % KD-TREE
                [index,nodenumber] = kd_closestpointfast(model.kdtree,desc);
                
                % Detector matrix update
                if find(index == detector_codewords)
                    det_matrix(TD.y(i), TD.x(i)) = det_matrix(TD.y(i), TD.x(i)) + 1; % Assumes original x and y are available, can?
                end
                
                % Display purpose in MATLAB (not needed for actual
                % FPGA)
                temp_count_matrix(TD.y(i), TD.x(i)) = temp_count_matrix(TD.y(i), TD.x(i)) + 1;
                
                % Increment BoW histogram
                bow_histogram(index) = bow_histogram(index) + 1;
                
                desc_count = desc_count + 1;
                
            end
            
            if desc_count == event_chunk_period % Classify every 10k events or so
                count_sep = count_sep + 1;
                
                % Normalized BoW histogram                
                scores = svmmodel.w' * bow_histogram + svmmodel.b' * ones(1,size(bow_histogram,2)) ;
                [obj_cat(count_sep,1), obj_cat(count_sep,2)] = max(double(scores)); % storing for offline comparison
                if obj_cat(count_sep,2) == det_class  % Detect only when object is present in the FOV
                    
                    [detected_locs_y,detected_locs_x]  = find(det_matrix > max(det_matrix(:))-1);
                    
                    figure(1),
                    imshow(temp_count_matrix)
                    hold on
                    plot(ceil(mean(detected_locs_x)), ceil(mean(detected_locs_y)), 'g+', 'MarkerSize',30);
                    hold off
                    drawnow()
                    
                end
                test_label = [test_label class_i]; % only for offline comparison
                
                desc_count = 0; %reset
                bow_histogram = zeros(histopts.num_bins, 1);
                disp(count_sep);
                
                %reset detection matrix
                det_matrix = zeros(180,240);
                temp_count_matrix = det_matrix;
            end
        end
    end
end

error_locs=find(obj_cat(:,2)'~=test_label)
error_per=length(error_locs)
accuracy= 100*(numel(test_label)-  error_per)/numel(test_label)