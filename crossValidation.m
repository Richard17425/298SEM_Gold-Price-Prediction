clear; clc; close all;

%% 1. Read Data
filename = 'gold_prices.xlsx';
data = readtable(filename);

% Date、XAU/USD、XAU/EUR, SILVER Price
data.Time = datetime(data{:,1}, 'InputFormat', 'MM-dd-yyyy');
data.XAU_USD = data.("Var2"); % Gold price against US Dollar
data.XAU_EUR = data.("Var3"); % Gold price against EURO
data.SILVER = data.("Var4");  % Silver price against US Dollar
data.ETF = data.("Var5");    % Exchange-Traded Fund Price
data.SP500 = data.("Var6");   % S&P 500 index
% Convert to timetable
DataTimeTable = table2timetable(data(:, ["Time", "XAU_USD", "XAU_EUR","SILVER","ETF","SP500"])); % 

%% 2. Cross Validation Setup
window_size = 100;
test_size = 20;
num_folds = floor((height(DataTimeTable) - window_size - test_size) / test_size); % 交叉验证的轮数
numlags = 4;
predict_step = 2;

MAE_folds = zeros(num_folds, 1);
RMSE_folds = zeros(num_folds, 1);
MAPE_folds = zeros(num_folds, 1);
R2_folds = zeros(num_folds, 1);
seriesnames = ["XAU_USD" "XAU_EUR" "SILVER" "ETF" "SP500"];
PriorMdl = bayesvarm(numel(seriesnames), numlags, 'SeriesNames', seriesnames);
%% 3. Cross Validation - this take a long time
for fold = 1:num_folds
    % define dataset
    train_start = (fold - 1) * test_size + 1;
    train_end = train_start + window_size - 1;
    test_start = train_end + 1;
    test_end = test_start + test_size - 1;

    train_data = DataTimeTable(train_start:train_end, :);
    test_data = DataTimeTable(test_start:test_end, :);

    % train model

    PosteriorMdl = estimate(PriorMdl, train_data{:, seriesnames});

    fprintf('loop--> %.2f%%\n', (fold / num_folds) * 100);

    % Test model
    % [YMean, ~] = forecast(PosteriorMdl, height(test_data), train_data{:, seriesnames});
    YMean_total = [];

    for i = 0:predict_step:height(test_data)-predict_step
        current_data = DataTimeTable{train_start+i:train_end+i, :};
        [YMean, ~] = forecast(PosteriorMdl, predict_step, current_data);
        YMean_total = [YMean_total;YMean];
        % YCI_total = [YCI_total;YCI];
        fprintf('inside_loop--> %.2f%%\n', (i / (height(test_data)-predict_step)) * 100);
    end

    actual_values = test_data.XAU_USD;  % actual gold price
    predicted_values = YMean_total(:, 1);    % predicted gold price

    % calculate index
    [MAE_folds(fold), RMSE_folds(fold), MAPE_folds(fold), R2_folds(fold)] = evaluate_forecast(actual_values, predicted_values);
    
end

%% 4. Calculate Average Performance Index
avg_MAE = mean(MAE_folds);
avg_RMSE = mean(RMSE_folds);
avg_MAPE = mean(MAPE_folds);
avg_R2 = mean(R2_folds);

%% 5. Display cross validation results
fprintf('Cross-Validation Results (Average):\n');
fprintf('MAE: %.4f\n', avg_MAE);
fprintf('RMSE: %.4f\n', avg_RMSE);
fprintf('MAPE: %.4f%%\n', avg_MAPE);
fprintf('R²: %.4f\n', avg_R2);

%% 6. Visualizing cross-validation metrics
figure;
subplot(2, 2, 1);
plot(MAE_folds, '-o',LineWidth=1.5);
hold on
yline(avg_MAE, '--r', 'LineWidth', 1);
title('MAE for Each Fold');
xlabel('Fold');
ylabel('MAE');
legend('MAE','Average MAE')
grid on;

subplot(2, 2, 2);
plot(RMSE_folds, '-o',LineWidth=1.5);
hold on
yline(avg_RMSE, '--r', 'LineWidth', 1);
title('RMSE for Each Fold');
xlabel('Fold');
ylabel('RMSE');
legend('RMSE','Average RMSE')
grid on;

subplot(2, 2, 3);
plot(MAPE_folds, '-o',LineWidth=1.5);
hold on
yline(avg_MAPE, '--r', 'LineWidth', 1);
title('MAPE (%) for Each Fold');
xlabel('Fold');
ylabel('MAPE (%)');
legend('MAPE','Average MAPE')
grid on;

subplot(2, 2, 4);
plot(R2_folds, '-o',LineWidth=1.5);
hold on
yline(avg_R2, '--r', 'LineWidth', 1);
title('R² for Each Fold');
xlabel('Fold');
ylabel('R²');
legend('R²','Average R²')
grid on;

sgtitle('Cross Validation (window size = 100, test size = 20, fold = 57)');

%% compare for error
function [MAE, RMSE, MAPE, R2] = evaluate_forecast(actual, predicted)
    % Evaluate the accuracy of a prediction model using key metrics
    % Inputs:
    %   actual - vector of actual values (blue line)
    %   predicted - vector of predicted values (red line)
    % Outputs:
    %   MAE - Mean Absolute Error
    %   RMSE - Root Mean Squared Error
    %   MAPE - Mean Absolute Percentage Error
    %   R2 - Coefficient of Determination

    % Ensure input vectors have the same length
    if length(actual) ~= length(predicted)
        error('Error: The input vectors must have the same length.');
    end
    
    % Compute Mean Absolute Error (MAE)
    MAE = mean(abs(actual - predicted));

    % Compute Root Mean Squared Error (RMSE)
    RMSE = sqrt(mean((actual - predicted).^2));

    % Compute Mean Absolute Percentage Error (MAPE)
    MAPE = mean(abs((actual - predicted) ./ actual)) * 100;

    % Compute R-squared (R²)
    SS_res = sum((actual - predicted).^2);  % Residual sum of squares
    SS_tot = sum((actual - mean(actual)).^2); % Total sum of squares
    R2 = 1 - (SS_res / SS_tot);  % Coefficient of determination

    % Display results
    fprintf('Prediction Accuracy Metrics:\n');
    fprintf('MAE  = %.4f\n', MAE);
    fprintf('RMSE = %.4f\n', RMSE);
    fprintf('MAPE = %.2f%%\n', MAPE);
    fprintf('R²    = %.4f\n', R2);
end