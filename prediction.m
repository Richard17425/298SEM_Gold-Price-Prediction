clear; clc; close all;

%% 1. Read data
filename = 'gold_prices.xlsx';
data = readtable(filename);

% Date、XAU/USD、XAU/EUR, SILVER Price
data.Time = datetime(data{:,1}, 'InputFormat', 'MM-dd-yyyy');
data.XAU_USD = data.("Var2"); % Gold price against US Dollar
data.XAU_EUR = data.("Var3"); % Gold price against EURO
data.SILVER = data.("Var4");  % Silver price against US Dollar
data.ETF = data.("Var5");    % Exchange-Traded Fund Price
data.SP500 = data.("Var6");   % S&P 500 index

DataTimeTable = table2timetable(data(:, ["Time", "XAU_USD", "XAU_EUR","SILVER","ETF","SP500"])); % 

%% 2. Data Splitting
train_idx = DataTimeTable.Time < datetime('2024-12-19'); % Train set
test_idx = DataTimeTable.Time >= datetime('2024-12-19'); % Test set

train_data = DataTimeTable(train_idx, :);
test_data = DataTimeTable(test_idx, :);
test_dates = test_data.Time;

%% 3. Select the lag order (adjustable)
numlags = 4;

%% 4. Train BVAR Modle
seriesnames = ["XAU_USD" "XAU_EUR" "SILVER" "ETF" "SP500"];
PriorMdl = bayesvarm(numel(seriesnames), numlags, 'SeriesNames', seriesnames);
PosteriorMdl = estimate(PriorMdl, train_data{:, seriesnames});

%% 5. Predict Future Gold Price
predict_step = 2;
num_past = height(train_data);
num_periods = height(test_data);
YMean_total = [];
YCI_total = [];
    for i = 0:predict_step:num_periods-predict_step
        current_data = DataTimeTable{num_past-30+i:num_past+i, :};
        [YMean, YCI] = forecast(PosteriorMdl, predict_step, current_data);
        YMean_total = [YMean_total;YMean];
        YCI_total = [YCI_total;YCI];
        fprintf('loop--> %.2f%%\n', (i / num_periods) * 100);
    end


% Extract actual and predicted values
actual_values = test_data.XAU_USD;  % Actual gold prices (Blue Line)
predicted_values = YMean_total(:,1);  % Predicted prices (Red Line)

% Ensure both vectors are of the same length
min_length = min(length(actual_values), length(predicted_values));
actual_values = actual_values(1:min_length);
predicted_values = predicted_values(1:min_length);
% Call the evaluation function and print results
[MAE, RMSE, MAPE, R2] = evaluate_forecast(actual_values, predicted_values);

%% 6. Visualizing prediction results
figure;
hold on;
plot(test_dates, test_data.XAU_USD, 'b', 'LineWidth', 2); % Real gold price (USD)
plot(test_dates, YMean_total(:,1), 'r--', 'LineWidth', 2); % Predicted value

lower_bound = YMean_total(:,1) - YCI_total(:, 1); 
upper_bound = YMean_total(:,1) + YCI_total(:, 2); 
fill([test_dates; flipud(test_dates)], [lower_bound; flipud(upper_bound)], ...
    'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% fill([test_dates; flipud(test_dates)], [YCI(:,1,1); flipud(YCI(:,1,2))], ...
%     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
legend('Real XAU/USD', 'Predicted XAU/USD', '± 1.96σ (95% Confidence Interval)', 'Location', 'Best');
xlabel('Date');
ylabel('Gold Price (USD)');
title('Gold Price Forecasted by Bayesian VAR');
grid on;
hold off;


variance = ((YCI_total(:, 2) - YMean_total(:, 1)) / 2).^2;

% Calculate differential entropy
differential_entropy = 0.5 * log(2 * pi * exp(1) * variance);

figure;
plot(test_dates, differential_entropy, 'b', 'LineWidth', 2);
title('Differential Entropy of Forecast Distribution');
xlabel('Time');
ylabel('Entropy');
grid on;

%% Calculate Residuals

residuals = actual_values - predicted_values;

figure;

% (a) Histogram of residual distribution
subplot(2, 1, 1);
histogram(residuals, 30, 'Normalization', 'pdf', 'FaceColor', 'b', 'EdgeColor', 'k');
title('Residual Distribution');
xlabel('Residual');
ylabel('Density');
grid on;

% (b) Draw a normal distribution fitting curve for the residuals
hold on;
x = linspace(min(residuals), max(residuals), 100);
pd = fitdist(residuals, 'Normal');
y = pdf(pd, x);
plot(x, y, 'r-', 'LineWidth', 2);
legend('Histogram', 'Normal Fit');
hold off;

% Plotting the residual time series
subplot(2, 1, 2);
plot(test_dates(1:length(residuals)), residuals, 'k-', 'LineWidth', 1.5);
yline(0, '--r', 'LineWidth', 1.5);
title('Residual Time Series');
xlabel('Date');
ylabel('Residual');
grid on;

% Analyzing Autocorrelation（ACF）
figure;
autocorr(residuals, 'NumLags', 20);
title('Residual Autocorrelation (ACF)');

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


%% plot price data

% figure;
% 
% subplot(3, 2, [5, 6]);
% plot(data.Time, data.XAU_USD, 'LineWidth', 1.5);
% title('XAU/USD');
% xlabel('Time');
% ylabel('Price');
% grid on;
% 
% subplot(3, 2, 1);
% plot(data.Time, data.XAU_EUR, 'LineWidth', 1.5);
% title('XAU/EUR');
% xlabel('Time');
% ylabel('Price');
% grid on;
% 
% subplot(3, 2, 2);
% plot(data.Time, data.SILVER, 'LineWidth', 1.5);
% title('Silver Price');
% xlabel('Time');
% ylabel('Price');
% grid on;
% 
% % ETF
% subplot(3, 2, 3);
% plot(data.Time, data.ETF, 'LineWidth', 1.5);
% title('ETF');
% xlabel('Time');
% ylabel('Price');
% grid on;
% 
% % S&P 500
% subplot(3, 2, 4);
% plot(data.Time, data.SP500, 'LineWidth', 1.5);
% title('S&P 500');
% xlabel('Time');
% ylabel('Index');
% grid on;
% 
% sgtitle('Time Series of Financial Data');