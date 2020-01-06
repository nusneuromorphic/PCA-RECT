function tree = convert_tree_template( kdtree )
% This functions converts the kdtree to a friendly format for processing
% and usage for kd_closestpointfast

% Input kd-tree is of VLFEAT template


s = load('tree_template.mat'); % use functional load to keep track of variables
tree = s.tree;
clearvars s
splitval = round(kdtree.trees.nodes.splitThreshold);
splitdim = kdtree.trees.nodes.splitDimension;
left = kdtree.trees.nodes.lowerChild;
right = kdtree.trees.nodes.upperChild;
index = kdtree.trees.dataIndex; 

type1 = 'node';
type2 = 'leaf';

for i= 1:numel(splitval)
    
    if left(i) > 0 
        tree(i).type = type1;
        tree(i).left = left(i);
        tree(i).right = right(i);
        tree(i).index = 0; %denotes a node, a positive number denotes data point
        tree(i).splitval = splitval(i);
        tree(i).splitdim = splitdim(i);
    else
        tree(i).type = type2;
        tree(i).left = [];
        tree(i).right = [];
        tree(i).index = index(abs(left(i))); %denotes data point
        tree(i).splitval = 0;
        tree(i).splitdim = [];
    end
end

