data = readtable("\\thalassa\ProjectLibrary\901805_Coastal_Biogeochemical_Sensing\Wetlab_Sensor_Calibration\NanoFet\K0\06_29_26_id_16_19_20_11.csv");
data = data(300:end,:);
data.gliderfet_id = categorical(data.gliderfet_id);

%%
xVar = 'datetime';   % <- replace with the real column name
yVar = 'vrse';   % <- replace with the real column name

data = sortrows(data, xVar);          % so each line draws in X order
ids  = categories(data.gliderfet_id); % unique IDs (e.g. 16, 19, 20)

figure
hold on
for i = 1:numel(ids)
    rows = data.gliderfet_id == ids(i);
    plot(data.(xVar)(rows), data.(yVar)(rows), '-', 'DisplayName', char(ids(i)))
end

xlabel(xVar, 'Interpreter','none')    % 'none' stops underscores -> subscripts
ylabel(yVar, 'Interpreter','none')
legend('Location','best')