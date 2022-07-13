ccc;

files = {'123189', '123191', '123192'};

figure;
shapesn = {'rd--', 'bd--', 'gd--'};
shapeso = {'ro-', 'bd-', 'go-'};

for i = 1:length(files)
    data = load([files{i} '_vgg16/' files{i} '_vgg16.mat']);
    
    data.comments
    plot(data.test_stats_nt(:, 2), shapesn{i})
    hold on
    plot(0:15, [70.26; data.test_stats_ot(:, 2)], shapeso{i})
    
end

grid minor
xlabel('Epochs')
ylabel('Accuracy')