clear;
close all;

% Add VLFeat toolbox to MATLAB working path
run('vlfeat-0.9.17-bin/vlfeat-0.9.17/toolbox/vl_setup.m');
addpath(genpath('getNmnistDesc'));

% Put all the folders (classes) in a single folder
train_dataset_path = '../N-SOD Dataset/Train';

filenamesA = dir2(train_dataset_path);

num_classes=numel(filenamesA);
classnames = cell(1, num_classes);
for i=1:num_classes
    classnames{i}=filenamesA(i).name;
end
% create training labels later
train_label=[];

num_test= 2;
filename = 'traindata_binTD4cl_CYnewobs.dat';
testfiles_save = 'testfiles_binTD4cl_CYnewobs';

if exist(filename, 'file') ~= 2
    % Training and testing set filenames
    for i=1:num_classes
        
        folder_path = fullfile(train_dataset_path,classnames{i});
        subfolder_names = dir(folder_path);
        isdir_tags = [subfolder_names.isdir];
        subfolder_names = subfolder_names(~isdir_tags);
        subfolder_names = {subfolder_names.name};
        
        randomize_files{i} = randperm(numel(subfolder_names));
        num_train = numel(randomize_files{i});
        index= [];
        
        for ii = randomize_files{i}
            filepath = fullfile(folder_path, subfolder_names{ii});
            img = read_linux(filepath);
            if size(img.x,2) > 5000
                index = [index ii];
            end
            if numel(index) > 80
                break
            end
        end
        disp(i);
        train_filenames{i} = vl_colsubset(subfolder_names(index),num_train, 'beginning');
        test_filenames{i} =  vl_colsubset(subfolder_names(index),num_test, 'beginning');
    end
    
    fid = fopen(filename, 'w');
    
    % Write into file
    for i = 1: numel(train_filenames)
        classi = train_filenames{i};
        for j = 1: numel(classi)
            classi_objectj = classi{j};
            fprintf(fid, '%s\n', classi_objectj );
        end
    end
    save(testfiles_save,'test_filenames');
end
fid = fopen(filename, 'r');
A = textscan(fid,'%s', 'Delimiter','\n');
all_filenames= A{1};
fclose(fid);

load(testfiles_save);

param.descsize = 7; % denotes a N by N region sampled from the subsampled count matrix
param.queuesize = 5000;
param.countmatsubsamp = 2; % a 2 by 2 cell region for the count matrix
param.minNumEvents = 200; % wait for this many events before getting desc from queue

disp(param);

% TD Noise Filter
% Assumes Garrick's AER functions are available (www.garrickorchard.com)
us_time_filter = 5e3; % in micro-seconds

% Debug purpose --
training_desc_done = 0; % for debug purposes only
svmtraining_done = 0;
testing_desc_done = 0;

% The training phase is here so that it is easy to change parameters
% and check the code of the FPGA implmentation. The testing stage will be
% made modular like how the FPGA implementation is intended to be, fixed
% precision and memory constraints closely followed.

count = 0;
train_desc_savefolder = './ECtraindata_FASTWithDet/';
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
            filename = all_filenames{count};
            filepath = fullfile(train_dataset_path,classi_name, filename);
            
            TD = read_linux(filepath);
            TD = FilterTD(TD, us_time_filter);
            
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
    
    save('./NDEMO4desc_7x7subsamp2x2_ustime5e3.mat',...
        'train_label','trainimage_sizes','-v7.3');
else
    disp('Loading descrs...');
    load('./NDEMO4desc_7x7subsamp2x2_ustime5e3.mat');
end

histopts.num_bins = 150; % codebook size
histopts.numSpatialX = [1] ; % ignore but required for code, if [1 2] performs spatial pyramid matching (not needed)
histopts.numSpatialY = [1] ; % ignore

% Initialize detection vector
det_vector = zeros(histopts.num_bins, 1);
det_class = find(strcmp(classnames, 'Thumper')); % the integer denoting the Thumper class

% SVM params
svmOpts.C = 10 ;
svmOpts.biasMultiplier = 1 ;

% Build the codebook
clearvars model net
model_str_stringname = 'modelTD4cl_DEMO_FASTWithDet';
try
    load(['./ECtrainmodels/' model_str_stringname ...
        num2str(histopts.num_bins) num2str(param.countmatsubsamp) num2str(param.descsize)]);
    model_done = 1;
catch
    model_done = 0;
end

if model_done == 0
    [train_data, loctrain_label, new_train_label] = readDescs(train_desc_savefolder,8,'nonorm');
    [model.vocab, model.assoc] = vl_kmeans(vl_colsubset(single([train_data.desc]), 4e6), histopts.num_bins, 'verbose','algorithm', 'ANN') ;
    model.kdtree = vl_kdtreebuild(model.vocab, 'Distance','L1') ;
    model.vocab = double(model.vocab);
    mkdir('./ECtrainmodels/');
    save(['./ECtrainmodels/' model_str_stringname ...
        num2str(histopts.num_bins) num2str(param.countmatsubsamp) num2str(param.descsize)],'model');
end

% For possbile integer comparisons on FPGA
model.kdtree.trees.nodes.splitThreshold = floor(model.kdtree.trees.nodes.splitThreshold);

% Get hists
for repeat= 1
    clearvars hists*
    
    train_label = [];
    count_sep = 0;
    count = 0;
    event_chunk_period = 1e4; % get a histogram for classification every 10k events, same for testing
    if svmtraining_done == 0
        for class_i=1:num_classes
            classname=classnames{class_i};
            folder_path = fullfile(train_dataset_path,classname);
            subfolder_names = dir(folder_path);
            isdir_tags = [subfolder_names.isdir];
            subfolder_names = subfolder_names(~isdir_tags);
            subfolder_names = {subfolder_names.name};
            for j=1:numel(subfolder_names)
                count = count + 1;
                filename = all_filenames{count};
                
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
                    hists{count_sep} = get_imagehist(model,trainimage_sizes(count,:),...
                        double(frames), double(desc), histopts, 'nonorm');
                    
                    % Evaluate the codewords assigned to the Thumper class
                    if class_i == det_class
                        det_vector = det_vector + hists{count_sep};
                    elseif class_i == 1 || class_i == 4 % Background class
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
        
        detector_codewords = find(det_vector > 0 );
        
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
        save('svvmodel_DEMO_FPGAver2_150codebokNONORM_det.mat','svmmodel','detector_codewords');
        
    else
        load('svvmodel_DEMO_FPGAver2_150codebokNONORM_det.mat');
    end
    
    %% -------------------------------------------------------------------------
    % -------------------------------------------------------------------------
    % -------------------------------------------------------------------------
    % -------------------------------------------------------------------------
    % testing stage
    count = 0;
    test_label = [];
    test_desc_savefolder = './ECtestdata_FASTWithDet/';
    mkdir(test_desc_savefolder);
    if testing_desc_done == 0
        for class_i=1:num_classes
            classi_name = cell2mat(classnames(class_i));
            test_classi = test_filenames{class_i};
            mkdir(fullfile(test_desc_savefolder, classi_name));
            for image_classi = 1:numel(test_classi)
                count = count + 1;
                filename = test_classi{image_classi};
                filepath = fullfile(train_dataset_path,classi_name, filename);
                
                savepath_file = fullfile(test_desc_savefolder, classi_name, filename(1:end-4));
                
                TD = read_linux(filepath);
                TD = FilterTD(TD, us_time_filter);
                
                testimage_sizes(count,:) = [max(TD.y) max(TD.x)];
                
                try
                    load(savepath_file);
                catch
                    [frames, desc] = getDescs_RECT_FPGA(TD, param);
                    disp(savepath_file);
                    save(savepath_file,'frames','desc');
                end
                
                test_label = [test_label class_i];
                disp(count);
                
            end
        end
        testing_desc_done =1;
        save('./NDEMO4testdesc_7x7subsamp2x2_ustime5e3.mat',...
            'test_label','testimage_sizes','-v7.3');
    else
        disp('Loading test descrs...'); % nneds modification of code if desc_done=0
        load('./NDEMO4testdesc_7x7subsamp2x2_ustime5e3.mat');
    end
    
    %% ---------------------------------------------------------------------------
    %---------------------------------------------------------------------------
    % Testing phase
    % Detection matrix init
    det_matrix = zeros(trainimage_sizes(1,1),trainimage_sizes(1,2));  % trainimage_sizes is in [y x] order already
    temp_count_matrix = det_matrix;
    
    count = 0;
    count_sep =0;
    save_for_majvot = [];
    test_label = [];
    for class_i=1:num_classes
        classi_name = cell2mat(classnames(class_i));
        test_classi = test_filenames{class_i};
        for image_classi = 1:numel(test_classi)
            count = count + 1;
            filename = test_classi{image_classi};
            filepath = fullfile(test_desc_savefolder, classi_name, [filename(1:end-4) '.mat']);
            
            [test_data, loctest_label, testlabel] = readDescs(filepath, [], 'nonorm');
            
            % Read original TD for detection matrix update
            filepath = fullfile(train_dataset_path,classi_name, filename);
            TD = read_linux(filepath);
            TD = FilterTD(TD, us_time_filter);
            
            start_chunk = 1;
            end_chunk = event_chunk_period;
            for event_c = 1:ceil(size(test_data.frames,2)/event_chunk_period)
                count_sep= count_sep + 1;
                try
                    desc = test_data.desc(:, start_chunk:end_chunk);
                    frames = test_data.frames(:,start_chunk:end_chunk);
                catch
                    desc = test_data.desc(:, start_chunk:end);
                    frames = test_data.frames(:,start_chunk:end);
                end
                
                
                hists = get_imagehist(model,testimage_sizes(count,:),...
                    double(frames), double(desc), histopts, 'nonorm');
                psix = hists;
                
                %SVM Classification
                scores = svmmodel.w' * psix + svmmodel.b' * ones(1,size(psix,2)) ;
                [obj_cat(count_sep,1), obj_cat(count_sep,2)] = max(scores);
                
                if obj_cat(count_sep,2) == det_class
                    binsa = double(vl_kdtreequery(model.kdtree, double(model.vocab), ...
                        double(desc), 'MaxComparisons', 15)) ;
                    
                    [ind, ~]=ismember(binsa, detector_codewords);
                    index_desc = find(ind == 1);
                    
                    detected_locations_x = TD.x(index_desc + (event_c - 1) * event_chunk_period + param.minNumEvents);
                    detected_locations_y = TD.y(index_desc + (event_c - 1) * event_chunk_period + param.minNumEvents);
                    
                    linear_indices = sub2ind(size(det_matrix), detected_locations_y, detected_locations_x);
                    det_matrix(linear_indices) = det_matrix(linear_indices) + 1;
                    
                    % Display purposes only   % Fun fun fun fun looking forward to the detection
                    try
                        all_locations_x = TD.x(1 + (event_c - 1) * event_chunk_period + param.minNumEvents: event_chunk_period + (event_c - 1) * event_chunk_period + param.minNumEvents);
                        all_locations_y = TD.y(1 + (event_c - 1) * event_chunk_period + param.minNumEvents: event_chunk_period + (event_c - 1) * event_chunk_period + param.minNumEvents);
                    catch
                        all_locations_x = TD.x(1 + (event_c - 1) * event_chunk_period + param.minNumEvents: end);
                        all_locations_y = TD.y(1 + (event_c - 1) * event_chunk_period + param.minNumEvents: end);
                    end
                    all_linear_indices = sub2ind(size(det_matrix), all_locations_y, all_locations_x);
                    temp_count_matrix(all_linear_indices) = temp_count_matrix(all_linear_indices) + 1;
                    
                    figure(1),
                    imshow(temp_count_matrix)
                    hold on
                    plot(ceil(mean(detected_locations_x)), ceil(mean(detected_locations_y)), 'g+', 'MarkerSize',30);
                    hold off
                    drawnow()
                    
                    %Reset count matrix and detection matrix
                    det_matrix = zeros(trainimage_sizes(1,1),trainimage_sizes(1,2));  % trainimage_sizes is in [y x] order already
                    temp_count_matrix = det_matrix;
                end
                test_label = [test_label class_i];
                start_chunk = end_chunk + 1;
                end_chunk = end_chunk + event_chunk_period;
                
                if end_chunk > size(test_data.frames,2)
                    end_chunk = size(test_data.frames,2);
                end
            end
            save_for_majvot(count) = count_sep;
            disp(count);
            
        end
    end
    
    error_locs=find(obj_cat(:,2)'~=test_label)
    error_per=length(error_locs)
    accuracy(repeat)= 100*(numel(test_label)-  error_per)/numel(test_label)
end