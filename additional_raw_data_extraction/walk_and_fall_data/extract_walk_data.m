% Create a subdirectory to save the .mat files
subdirectory = 'extracted_walking_data';
if ~isfolder(subdirectory)
    mkdir(subdirectory);
end

% Get a list of all .mat files in the current directory
fileList = dir('*.mat');

pattern = '.*_.*_(\d+)\.mat';

% Iterate over each .mat file
unique_int = 0;
for i = 1:numel(fileList)
    filename = fileList(i).name;
    class = regexp(filename, pattern, "tokens");
    class = class{1}{1};

    % Load the .mat file
    data = load(filename);
    
    % Access the loaded variables from the .mat file
    raw_data = data.hil_resha_aligned;
    labels = data.lbl_out(1:length(raw_data)); % label vector is longer sometimes
    labels(labels ~= 1) = 0;
    
    % Cheeky lil hack
    pop_first = false;
    if labels(1) == 0
        labels(1) = 1;
        pop_first = true;
    end
    
    ne0 = find(labels ~= 0);                            % Nonzero Elements
    ix0 = unique([ne0(1) ne0(diff([0 ne0])>1)]);        % Segment Start Indices
    ix1 = ne0([find(diff([0 ne0])>1)-1 length(ne0)]);   % Segment End Indices
    
    % Cheeky lil hack
    if pop_first
        ix0 = ix0(2:end);
        ix1 = ix1(2:end);
    end
    
    % Split the data
    for k1 = 1:length(ix0)
        section{k1} = raw_data(:, ix0(k1):ix1(k1), :);
    end

    % Save each matrix to a separate .mat file
    for j = 1:numel(section)
        hil_resha_aligned = section{j};
        if size(hil_resha_aligned, 2) > 200 % low estimate, will be filtered in preprocessing
            filename = fullfile(subdirectory, sprintf('%d_WandF_extr_wal_%s.mat', unique_int, class));
            save(filename, 'hil_resha_aligned');
            unique_int = unique_int + 1;
        end
    end
end
    
