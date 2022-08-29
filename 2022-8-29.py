import numpy as np
from mindspore import dataset as ds
from mindspore import nn
from mindspore import Model
from mindspore.train.callback import LossMonitor, Callback, TimeMonitor

def scaling_window(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length):
        _x = data[i:(i+seq_length), :]
        _y = data[i+seq_length, :]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

class MyDataSet:
    def __init__(self, data, label):
        self.data = data.astype(np.float32)
        self.label = label.astype(np.float32)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class LSTM(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_len):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=False,
                            dropout=0.02)

        self.linear = nn.Dense(in_channels=self.hidden_size,
                               out_channels=self.output_size,
                               weight_init='normal',
                               bias_init='normal')

    def construct(self, inputs):
        lstm_out, _ = self.lstm(inputs.view(len(inputs), self.seq_len, -1), None)
        last_time_step = lstm_out.view(self.seq_len, len(inputs), self.hidden_size)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred

class Valid_Callback(Callback):
    def __init__(self, model, ds):
        self.model = model
        self.data = ds
        self.predict_interval = 2
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            result = self.model.eval(self.data)
            print(result)


total_data = np.loadtxt(r'C:\Users\dluth\Desktop\用于训练的数据\PCA-result.txt') # 所有用于训练和测试的数据

train_data = total_data[0:60, :] # 60条数据用于训练
test_data = total_data[60:, :] # 60条数据用于测试

X_train, y_train = scaling_window(train_data, seq_length=3) # 训练的数据
X_test, y_test = scaling_window(test_data, seq_length=3) # 测试的数据

dataset_generator_train = MyDataSet(data=X_train, label=y_train) # 用于训练的数据集
dataset_train = ds.GeneratorDataset(dataset_generator_train, ["data_train", "label_train"], shuffle=False)

dataset_generator_valid = MyDataSet(data=X_test, label=y_test) # 用于测试的数据集
dataset_valid = ds.GeneratorDataset(dataset_generator_valid, ["data_valid", "label_valid"], shuffle=False)

dataset_train = dataset_train.batch(57)
dataset_valid = dataset_valid.batch(17)

lstm_net = LSTM(input_size=2, hidden_size=20, num_layers=2, output_size=2, seq_len=3) # 实例化LSTM网络
loss_fun = nn.MSELoss() # 损失函数
opt = nn.Adam(params=lstm_net.trainable_params(), learning_rate=0.001) # 优化器

model = Model(network=lstm_net, loss_fn=loss_fun, optimizer=opt, metrics={"mae"})

valid = Valid_Callback(model=model, ds=dataset_valid)
model.train(1000, dataset_train, callbacks=[LossMonitor(per_print_times=1), valid, TimeMonitor()])