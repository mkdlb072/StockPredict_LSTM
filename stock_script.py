import stock_lib
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Programme initialisation
stock_symbol = 'NVDA'
delta_time = 1095
data_window = 100
predict_window = 100
epoch = 10
flg_train_model = 1
flg_lstm_algo = 1
dev_env = 'NB'

# Get stock and earning data from yFinance
start_date = datetime.now() - timedelta(days=delta_time)
end_date = datetime.now()
raw_data = stock_lib.get_stock_data(stock_symbol, start_date, end_date)
np_data = raw_data.to_numpy()
earning_table, earning_next, earning_delta = stock_lib.get_earning_data(stock_symbol)

if flg_lstm_algo == 0:
    # Split the data for training and testing
    x, y = stock_lib.prepare_data(np_data, data_window)
    (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3)
    # Scaling the training data
    (x_train, x_test, y_train, y_test, sc) = stock_lib.minmaxscale_3d(x_train, x_test, y_train, y_test)
    # Train the model
    model = stock_lib.define_lstm_dnn_model(x_train)
    model = stock_lib.train_save_model(model, x_train, y_train, x_test, y_test, epoch, flg_train_model, stock_symbol, dev_env)
    predicted_p = stock_lib.get_past_prediction(model, np_data, predict_window, data_window, sc)
    predicted_f = stock_lib.get_future_prediction(model, np_data, predict_window, data_window, sc)
    print(predicted_p[:, 0])

elif flg_lstm_algo == 1:
    # Split the data for lstm_cat training and testing
    x1, x2, x3, y = stock_lib.prepare_data_3d(np_data, predict_window)
    (x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test) = stock_lib.train_test_split_3d(x1, x2,
                                                                                                               x3, y)
    # Scaling the training data
    (x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test, sc) = stock_lib.minmaxscale_4d(x1_train,
                                                                                                              x1_test,
                                                                                                              x2_train,
                                                                                                              x2_test,
                                                                                                              x3_train,
                                                                                                              x3_test,
                                                                                                              y_train,
                                                                                                              y_test)
    # Train the model
    model = stock_lib.define_lstm_cat_model(x1_train, x2_train, x3_train)
    model.summary()
    model = stock_lib.train_save_model(model, [x1_train, x2_train, x3_train], y_train, [x1_test, x2_test, x3_test],
                                       y_test, epoch, flg_train_model, stock_symbol, dev_env)
    predicted_p = stock_lib.get_past_prediction_3d(model, np_data, predict_window, sc)
    predicted_f = stock_lib.get_future_prediction_3d(model, np_data, predict_window, sc)
    print(predicted_p[:, 0])

elif flg_lstm_algo == 2:
    # Split the data for training and testing
    x, y = stock_lib.prepare_data(np_data, data_window)
    (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3)
    # Scaling the training data
    (x_train, x_test, y_train, y_test, sc) = stock_lib.minmaxscale_3d(x_train, x_test, y_train, y_test)
    # Train the model
    model = stock_lib.define_lstm_cnn_model(x_train)
    model.summary()
    model = stock_lib.train_save_model(model, x_train, y_train, x_test, y_test, epoch, flg_train_model, stock_symbol, dev_env)
    predicted_p = stock_lib.get_past_prediction(model, np_data, predict_window, data_window, sc)
    predicted_f = stock_lib.get_future_prediction(model, np_data, predict_window, data_window, sc)
    print(predicted_p[:, 0])

# # Predict future stock values
# predicted_p = stock_lib.get_past_prediction(model, np_data, predict_window, data_window, sc)
# predicted_f = stock_lib.get_future_prediction(model, np_data, predict_window, data_window, sc)
# print(predicted_p[:, 0])

# Plot the results
stock_lib.plot_prediction(stock_symbol, raw_data, predict_window, predicted_p, predicted_f, earning_delta)
stock_lib.save_graph(end_date, stock_symbol, flg_lstm_algo, dev_env)
