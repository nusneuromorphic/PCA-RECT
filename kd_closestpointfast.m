function [index_vals,final_node] = kd_closestpointfast(tree,point,node_number)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUTS
% tree        --- the cell array that contains the tree
% point       --- the point of interest
% none_number --- Internal Variable, Donot Define

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUTS
% index_vals  --- the index value in the original matrix that was used to
%                 build the tree
% final_node  --- the node that contains the closest vector

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Donot define the node_number variable -- it is used for internal
% referencing

% Bharath Ramesh 26/2/2018


% Initialize the global variable
global tree_cell;
global safety_check;

if(nargin==2)
    safety_check=0;
    node_number=1;
    tree_cell=tree;
    final_node=node_number;
    clear tree;
end

if (isempty(safety_check))
    error ('Incorrect number of input variables ... please check ');
end

% if the current node is a leaf then output its results
if(strcmp(tree_cell(node_number).type,'leaf'))
    index_vals=tree_cell(node_number).index;
    final_node=node_number;
    clear global tree_cell
    clear global safety_check;
    return;
end


% if the current node is not a leaf
% check to see if the point is to the left of the split dimension
% if it is to the left then recurse to the left
if (point(tree_cell(node_number).splitdim)<=tree_cell(node_number).splitval)
    if (isempty(tree_cell(node_number).left))
        % incase the left node is empty, then output current results
        index_vals=tree_cell(node_number).index;
        final_node=node_number;
        return;
    else
        [index_vals,final_node]=kd_closestpointfast(0,point,tree_cell(node_number).left);
    end
else
    % as the point is to the right of the split dimension
    % recurse to the right
    if (isempty(tree_cell(node_number).right))
        % incase the right node is empty, then output current results
        index_vals=tree_cell(node_number).index;
        final_node=node_number;
        return;
    else
        [index_vals,final_node]=kd_closestpointfast(0,point,tree_cell(node_number).right);
    end

end
