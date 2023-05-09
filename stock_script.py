import stock_lib
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Programme initialisation
stock_symbol = 'AAPL'
delta_time = 1095
data_window = 100
predict_window = 100
epoch = 1000

# Get stock and earning data from yFinance
start_date = datetime.now() - timedelta(days=delta_time)
end_date = datetime.now()
raw_data = stock_lib.get_stock_data(stock_symbol, start_date, end_date)
np_data = raw_data.to_numpy()
earning_table, earning_next, earning_delta = stock_lib.get_earning_data(stock_symbol)

# Split the data for training and testing
x, y = stock_lib.prepare_data(np_data, data_window)
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3)

# Split the data for lstm_cat training and testing
# x1, x2, x3, y1 = stock_lib.prepare_data_3x(np_data, predict_window)
# (x1_train, x1_test, y1_train, y1_test) = train_test_split(x, y, test_size=0.2)
# (x2_train, x2_test, y1_train, y1_test) = train_test_split(x, y, test_size=0.2)
# (x3_train, x3_test, y1_train, y1_test) = train_test_split(x, y, test_size=0.2)

# Scaling the training data
(x_train, x_test, y_train, y_test, sc) = stock_lib.minmaxscale_3d(x_train, x_test, y_train, y_test)

# Train the model
data_shape = (x.shape[1], x.shape[2])
# model = stock_lib.define_lstm_cnn_model(data_shape)
model = stock_lib.define_lstm_dnn_model(data_shape)
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=epoch, batch_size=32, validation_data=(x_test, y_test), verbose=2)

# Predict future stock values
predicted_p = stock_lib.get_past_prediction(model, np_data, predict_window, data_window, sc)
predicted_f = stock_lib.get_future_prediction(model, np_data, predict_window, data_window, sc)
print(predicted_p[:, 0])

# Plot the results
stock_lib.plot_prediction(stock_symbol, raw_data, predict_window, predicted_p, predicted_f, earning_delta)
