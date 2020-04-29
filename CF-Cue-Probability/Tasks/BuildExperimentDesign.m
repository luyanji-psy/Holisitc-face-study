function design = BuildExperimentDesign( expConditions, sortCondition )

%% create a factorial design for experiment and randomize, but order by sortCondition.

% Does a full-factorial design, similar to StatsToolbox function "fullfact",
% using the experiment conditions given in
% the cell array expConditions. expConditions is Nx2, where N is the number of levels
% with condition names in the first column and possible values for
% each condition in the second column, e.g.:

% % expConditions = {...
% %     'primeShapeLeft', 1:2 ; ...
% %     'primeShapeRight', 1:2 ; ...
% %     'primeDuration', 2/60 ; ...
% %     'primeMaskInterval', [2,4,6]/60 ; ...
% %     'blockNumber', 1:4 ...
% %     };

% Function returns a struct array with number of structs corresponding to
% number of trials. Each trial is a unique combination of possible 
% condition values.

% These unique trials appearing the output struct are in a random order.
% Trials can be balanced by blocks, or another condition, but adding a
% "sortCondition" string argument, which should correspond to a single
% condition name in expConditions.

clear expDesign levelsPerCondition;
levelsPerCondition = cellfun(@length,expConditions(:,2)); % cellfun applies a function to each cell of a cell array. here, computes the length of each cell in expConditions,
                                                          % namely, the number of levels in each expCondition
                                                         
%% check formatting of expConditions
if ~iscell(expConditions) || size(expConditions,2)~=2 || ~iscellstr(expConditions(:,1))
    error('expConditions was wrong type, wrong size, or malformed.');
end

%% is sortCondition given?
if nargin == 1
    sortCondition = []; 
elseif ~any(strcmpi(expConditions(:,1),sortCondition))  % any: True if any element of a vector is a nonzero number or is logical 1 (True)
    error(['Couldn''t find a condition in ''expConditions'' to match requested sortCondition, ''' sortCondition '''.']);
end
    
%% below code stolen from "fullfact" function in Statistics Toolbox.
ssize = prod(levelsPerCondition); % prod: Product of elements; ssize computes the total numbers of condition levle combinations
ncycles = ssize;
cols = length(levelsPerCondition);  % number of Conditions

designFF = zeros(ssize,cols,class(levelsPerCondition)); % create a matrix of zeros with ssize Rows and cols Columns

for k = 1:cols
   settings = (1:levelsPerCondition(k));                % settings for kth factor
   nreps = ssize./ncycles;                  % repeats of consecutive values
   ncycles = ncycles./levelsPerCondition(k);            % repeats of sequence
   settings = settings(ones(1,nreps),:);    % repeat each value nreps times
   settings = settings(:);                  % fold into a column
   settings = settings(:,ones(1,ncycles));  % repeat sequence to fill the array
   designFF(:,k) = settings(:);
end

clear ssize ncycles cols settings nreps

%% Raw full factorial is now 'designFF'.                          % It is FULL factorial design, not suitable for unbalanced cue probability in this experiment! like 25% top cue and 75% bottom cue
% randomize and sort by sortCondition.
designFF = sortrows( designFF(randperm(size(designFF,1)),:) ,...  % randperm: Random permutation
    find(strcmpi(expConditions(:,1),sortCondition)) );  % strcmpi: Compare strings or character vectors ignoring case.
                                                        % find: Find indices of nonzero elements.
                                                        % sortrows: Sort rows of a matrix. sortrows(A,COL) sorts the mtatrix according to the columns specified by the vector COL.
                                                        
cellDesign = cell(size(designFF)); % cell: Create cell array. cell(N) is an N-by-N cell array of empty metrices.
for iCol = 1:size(designFF,2)
    for iRow = 1:size(designFF,1)
        cellDesign{iRow,iCol} = expConditions{iCol,2}(designFF(iRow,iCol));  % designFF is a matrix of condition levels (e.g., 1,2,3); cellDesign is a matrix of condition values (e.g., lelvel1=0.03, level2=0.06, level3=0.1).
    end
end

design = cell2struct( cellDesign , expConditions(:,1), 2); % cell2struct: Convert cell array to structure array
%S = cell2struct(C,FIELDS,DIM) converts the cell array C into the structure S by folding the dimension DIM of C into fields of S. 

clear designFF cellDesign

end

