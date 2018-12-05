import numpy as np

data_num = 30
data_length = 40
hidden_unit = 20

num_iteration = 50
k = 0.00001

data = np.random.rand(data_num, data_length)

h_w = np.random.rand(data_length, hidden_unit)
h_b = np.random.rand(hidden_unit)

o_w = np.random.rand(hidden_unit, data_length)
o_b = np.random.rand(data_length)

for _ in range(num_iteration):
    batch = data[:]
    h_o = batch.dot(h_w) + h_b
    y_predict = h_o.dot(o_w) + o_b
    y_true = batch

    loss = np.mean(np.sum((y_predict - y_true) ** 2, axis=0))
    d_l = k * 2 * (y_predict - y_true)
    d_o_w = h_o.transpose().dot(d_l)
    d_o_b = np.mean(np.ones(1) * d_l, axis=0)

    # d_h_o = np.sum(o_w, axis=0)
    d_h_w = d_l.dot(o_w.transpose()).transpose().dot(batch).transpose()
    d_h_b = np.mean(d_l.dot(o_w.transpose()), axis=0)

    h_w -= d_h_w
    h_b -= d_h_b

    o_w -= d_o_w
    o_b -= d_o_b
    print(loss)
