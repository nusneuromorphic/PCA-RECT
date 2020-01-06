% Code for testing on N-Caltech101 using Event Signatures
clear; 
close all; 

% Add VLFeat toolbox to MATLAB working path
run('vlfeat-0.9.17-bin/vlfeat-0.9.17/toolbox/vl_setup.m');
addpath(genpath('getNmnistDesc'));

% Put all the folders (classes) in a single folder
train_dataset_path = '/hpctmp2/tslrame/Caltech101';

%load('filenamesA');
filenamesA = dir2(train_dataset_path);
%filenamesA(1:2) = []; % remove '.' and '..' folders

num_classes=numel(filenamesA);
for i=1:num_classes
    classnames{i}=filenamesA(i).name;
end
%remove BG class
classnames = classnames(2:end);
num_classes = num_classes - 1;
% create training labels later 
train_label=[];

num_train = 30;
num_test= 50; 
filename = 'traindata_binTD100cl.dat';
testfiles_save = 'testfiles_caltech100_binTD';

if exist(filename, 'file') ~= 2
    % Training and testing set filenames
    for i=1:num_classes

        folder_path = fullfile(train_dataset_path,classnames{i});
        subfolder_names = dir(folder_path);
        isdir_tags = [subfolder_names.isdir];
        subfolder_names = subfolder_names(~isdir_tags);
        subfolder_names = {subfolder_names.name};
        
        randomize_files{i} = randperm(numel(subfolder_names));
        index= [];
    
            for ii = randomize_files{i}
                filepath = fullfile(folder_path, subfolder_names{ii});
                img = Read_Ndataset(filepath);
               if size(img.x,1) > 5000
                index = [index ii];
               end
               if numel(index) > 80
                   break
               end
            end
   disp(i);
            train_filenames{i} = vl_colsubset(subfolder_names(index),num_train, 'beginning');
            test_filenames{i} =  vl_colsubset(subfolder_names(index(1+num_train:end)),num_test, 'beginning');

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

param.rmax = 10;
% param.bins = 7; % no of rings
param.binres = 2.5; % no of wedges
param.minNumEvents = 200;
%param.maxNumEvents = 10000;
%loctrain_label = [];
count = 0;

disp(param); 
loctrain_strlabel=[];

%image_sizes = [ 34 34]; % Refer to N-MNISpolarT paper for the padding

% TD Noise Filter 
% Assumes Garrick's AER functions are available (www.garrickorchard.com)
us_time_filter = 200e3; % in micro-seconds
us_time_filter_alt = 200e3;

% Debug purpose -- 
training_desc_done = 1;
testing_desc_done =1;

train_desc_savefolder = '/hpctmp2/tslrame/ECrecttraindata_rmax10_crop/';
annotation_folder = '/hpctmp2/tslrame/Caltech101_annotations/Caltech101_annotations/';
annot_offset = 7; % in pixels
mkdir(train_desc_savefolder);
if training_desc_done == 0
    for class_i=1:num_classes
        classi_name = cell2mat(classnames(class_i));
        mkdir(fullfile(train_desc_savefolder, classi_name));
        for train_classi = 1:num_train
           count = (class_i - 1) * num_train + train_classi; 
           filename = all_filenames{count};
           filepath = fullfile(train_dataset_path,classi_name, filename);

        TD = Read_Ndataset(filepath);
        TD = stabilize(TD);
        TD = filter_TD(TD, us_time_filter, us_time_filter_alt);
        
        string_filenames = strsplit(filename,'_');
        string_filenames = string_filenames{2};
        annot_filename = ['annotation_' string_filenames];
        annot_filepath = fullfile(annotation_folder,classi_name, annot_filename);
        
        [box_contour, obj_contour] = Read_annotation(annot_filepath);
        top_left = [box_contour(1,1),box_contour(2,1)];
        pixel_box_size = [box_contour(1,2) + annot_offset, box_contour(2,3) + annot_offset];
                
        ROI_X_indices = TD.x >= top_left(1) & TD.x <= (top_left(1) + pixel_box_size(1));
        ROI_Y_indices = TD.y >= top_left(2) & TD.y <= (top_left(2) + pixel_box_size(2));
        ROI_XY_indices = ROI_X_indices & ROI_Y_indices;
        %TD_noROI = RemoveNulls(TD, ROI_XY_indices);
        TD = RemoveNulls(TD, not(ROI_XY_indices));
        
        trainimage_sizes(count,:) = [max(TD.y) max(TD.x)];
        savepath_file = fullfile(train_desc_savefolder, classi_name, filename(1:end-4));
        
        try 
            load(savepath_file);
        catch 
            [frames, desc] = getDescs_rect(TD, param);

            savepath_file = fullfile(train_desc_savefolder, classi_name, filename(1:end-4));
            disp(savepath_file);
            save(savepath_file,'frames','desc');
        end
            
           train_label = [train_label class_i];
           disp(count);

        end
    end

 save('/hpctmp2/tslrame/NCAL100descrect_7x12_rmax10_rmin2_ustime200e3.mat',...
     'train_label','trainimage_sizes','-v7.3');

 poolobj = gcp('nocreate');
 delete(poolobj);

else
    disp('Loading descrs...'); % nneds modification of code if desc_done=0
load('/hpctmp2/tslrame/NCAL100descrect_7x12_rmax10_rmin2_ustime200e3.mat');
end

histopts.num_bins = 3000;
histopts.numSpatialX = [1 2 3] ;
histopts.numSpatialY = [1 2 3] ;

svmOpts.C = 10 ;
svmOpts.biasMultiplier = 1 ;

for pcadims = 5:5:80; 

 % Build the codebook
 clearvars model net
 model_str_stringname = 'modelTD100cl_CROPrectPCA_pro';    
try
	load(['/hpctmp2/tslrame/ECtrainmodels/' model_str_stringname ...
        num2str(histopts.num_bins) num2str(param.rmax) num2str(pcadims)]);
		model_done = 1;
catch
    model_done = 0;
end

if model_done == 0
  [train_data, loctrain_label, new_train_label] = readDescs(train_desc_savefolder,300);
  traindata = single([train_data.desc]);
          attr_means=mean(traindata');
        traindata=bsxfun(@minus,traindata',attr_means);

  [coeff, score, latent] = princomp(traindata,'econ');
%   save('/hpctmp/tslrame/tempPCADART_rectbinres2pt5.mat','coeff','score','latent','tsquare','-v7.3');
%   return
  traindata_afterpca = score(:,1:pcadims)';
  whos traindata_afterpca
    clearvars traindata score
  model.vocab = vl_kmeans(traindata_afterpca, histopts.num_bins, 'verbose','algorithm', 'ANN') ;
  model.kdtree = vl_kdtreebuild(model.vocab) ;
  model.vocab=double(model.vocab);
  save(['/hpctmp2/tslrame/ECtrainmodels/' model_str_stringname ...
        num2str(histopts.num_bins) num2str(param.rmax) num2str(pcadims)],'model', 'coeff', 'attr_means');
end


% Get hists
  for repeat= 1:10
      clearvars hists*
	
    train_label = [];
    for class_i=1:num_classes
        classname=classnames{class_i};
        for j=1:num_train
           count = (class_i - 1) * num_train + j; 
           filename = all_filenames{count};
           
            filepath=fullfile(train_desc_savefolder, classname, [filename(1:end-4) '.mat']);
            disp(filepath);
            [train_data, loctrain_label, label] = readDescs(filepath, []);
            traindata = train_data.desc;
            traindata=bsxfun(@minus,traindata',attr_means);
            traindata = traindata';
            traindata_afterpca = coeff(:,1:pcadims)' * traindata;
            hists{count} = get_imagehist(model,trainimage_sizes(count,:),...
                        double(train_data.frames), double(traindata_afterpca), histopts);
            train_label = [train_label class_i];
        end
    end

  hists = cat(2, hists{:}) ;
% 
% matrix_indices= sub2ind(size(magic(histoptsmsf.num_bins)),word_combs(:,1),word_combs(:,2));
% tic;
%   for i=1:length(train_label)
%            hists_msf{i}=get_imagehist_msfcontext(modelmsf,double(image_sizes(i,:)),...
%           double(train_data(i).frames), double(train_data(i).desc), histoptsmsf,matrix_indices);
%   end
%   toc
%   hists_msf = cat(2, hists_msf{:}) ;
 
%   hists=cat(1,hists,hists_msf);
  hists=cat(1,hists);
  hists = single(bsxfun(@rdivide,hists ,sum(hists)));
  hists = normc(hists); 
  %get feature map
psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5);


  %Pegasus SVM Calculation
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
   %save('svvmodel_CROPrect.mat','svmmodel');

    count = 0;
    test_label = []; 
    test_desc_savefolder = '/hpctmp2/tslrame/ECrecttestdata_rmax10_CROP/';
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
                
                TD = Read_Ndataset(filepath);
                TD = stabilize(TD);
                TD = filter_TD(TD, us_time_filter, us_time_filter_alt);

                string_filenames = strsplit(filename,'_');
                string_filenames = string_filenames{2};
                annot_filename = ['annotation_' string_filenames];
                annot_filepath = fullfile(annotation_folder,classi_name, annot_filename);

                [box_contour, obj_contour] = Read_annotation(annot_filepath);
                top_left = [box_contour(1,1),box_contour(2,1)];
                pixel_box_size = [box_contour(1,2) + annot_offset, box_contour(2,3) + annot_offset];

                ROI_X_indices = TD.x >= top_left(1) & TD.x <= (top_left(1) + pixel_box_size(1));
                ROI_Y_indices = TD.y >= top_left(2) & TD.y <= (top_left(2) + pixel_box_size(2));
                ROI_XY_indices = ROI_X_indices & ROI_Y_indices;
                %TD_noROI = RemoveNulls(TD, ROI_XY_indices);
                TD = RemoveNulls(TD, not(ROI_XY_indices));

                 testimage_sizes(count,:) = [max(TD.y) max(TD.x)];
                
                try 
                    load(savepath_file);
                    
                catch    
               
                    [frames, desc] = getDescs_rect(TD, param);
                     disp(savepath_file);
                    save(savepath_file,'frames','desc');
                end
                


                test_label = [test_label class_i];
                   disp(count);
                
            end
        end
         testing_desc_done =1;
         save('/hpctmp2/tslrame/NCAL100testdescrect_7x12_rmax10_rmin2_ustime200e3.mat',...
             'test_label','testimage_sizes','-v7.3');

         poolobj = gcp('nocreate');
         delete(poolobj);

    else
        disp('Loading test descrs...'); % nneds modification of code if desc_done=0
        load('/hpctmp2/tslrame/NCAL100testdescrect_7x12_rmax10_rmin2_ustime200e3.mat');
    end

    count = 0;
    for class_i=1:num_classes
            classi_name = cell2mat(classnames(class_i));
            test_classi = test_filenames{class_i};
            for image_classi = 1:numel(test_classi)
                count = count + 1;
                filename = test_classi{image_classi};
                filepath = fullfile(test_desc_savefolder, classi_name, [filename(1:end-4) '.mat']);

                [test_data, loctest_label, testlabel] = readDescs(filepath, []);
                testdata_befpca = test_data.desc;
                            testdata_befpca = bsxfun(@minus,testdata_befpca',attr_means);
            testdata_befpca = testdata_befpca';
                testdata_afterpca = coeff(:,1:pcadims)' * testdata_befpca;
                hists = get_imagehist(model,testimage_sizes(count,:),...
                        double(test_data.frames), double(testdata_afterpca), histopts);
                hists = normc(hists);     
                psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;
                %Pegasus Classification
                scores = svmmodel.w' * psix + svmmodel.b' * ones(1,size(psix,2)) ;
                [obj_cat(count,1), obj_cat(count,2)] = max(scores);
            
                 disp(count);
                
            end
     end



error_locs=find(obj_cat(:,2)'~=test_label)
error_per=length(error_locs)
accuracy(repeat)= 100*(numel(test_label)-  error_per)/numel(test_label)
weighted_acc(repeat) = get_weighted_acc(obj_cat, test_label)

save(['ECrecttestacc_and_results_CROPpca' num2str(pcadims) num2str(repeat) num2str(histopts.numSpatialX) num2str(histopts.num_bins) num2str(param.rmax)],'obj_cat', 'test_label', 'accuracy', 'weighted_acc'); 
  end
end

