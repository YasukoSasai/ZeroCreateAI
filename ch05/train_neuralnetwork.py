# ========= 学習の推移などを出します ========= (5.7.4)
# ========= two_layers_net の予測のprintをコメントアウトしてください(55,56,59,60行目) ==========
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layers_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10) #NNのインスタンス生成

# ========= ハイパーパラメータ ========= 
iters_num = 10000 #学習回数(パラメータ更新回数) iters = iterarion
train_size = x_train.shape[0] #x_trainデータ形状の0次元の数 = 60000枚
batch_size = 100 #batch_size = １回の学習で何枚分まとめて行うか
learning_rate = 0.1 #学習率

train_loss_list = [] #学習ごとの誤差(損失関数の結果)を格納するためのリスト
train_acc_list = [] #訓練データ(学習)における正確率を格納するためのリスト
test_acc_list = [] #テストデータ(テスト)における正確率を格納するためのリスト

# # ========= 重み・パラメータグラフ表示するとき ========= 
# w_update_list_0 = []
# w_update_list_4 = []
# w_update_list_9 = []

iter_per_epoch = max(train_size / batch_size, 1) #1エポックあたりの学習回数。600回。。

# ========= 学習フェーズ ========= 
for i in range (iters_num): #10000回繰り返し
    # ========= ミニバッチの取得 ========= 
    batch_mask = np.random.choice(train_size, batch_size) #train_sizeの中からbatch_sizeランダムでインデックスを配列で取り出す。
    x_batch = x_train[batch_mask] #bacth_mask番目のx_train(配列)をx_batchに代入。
    t_batch = t_train[batch_mask] #同様

    # ========= 勾配計算 ========= 
    #数値微分
    # grad = network.numerical_gradient(x_batch, t_batch)
    #誤差逆伝播法　高速！ 
    grad = network.gradient(x_batch, t_batch) 

    # ========= パラメータ更新 ========= 
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

        # # ========= 重みとパラメータのグラフを表示したい時(W2の一部) ========= 
        # weight_per_data = list(network.params['W2'])
        # weight_per_neuron = list(weight_per_data)

        # weight_first = weight_per_neuron[0]

        # weight_first_0 = weight_first[0]
        # weight_first_4 = weight_first[4]
        # weight_first_9 = weight_first[9]   

        # if key == 'W2':
        #     w_update_list_0.append(weight_first_0)
        #     w_update_list_4.append(weight_first_4)
        #     w_update_list_9.append(weight_first_9)

    # ========= 誤差の記録 ========= 
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

# ========= テストフェーズ =========     
#1エポックごとにテストデータで認識精度を計算　計算時間がかかるのでざっくり。
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# ************* グラフ化 ******************
# ========= 学習による誤差推移 ========= 
plt.plot(train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show() 

# ========= 訓練データとテストデータで認識精度をグラフ化(汎化性能を見るため) ========= 
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# ========= 損失関数とパラメータのグラフ ========= 
# plt.plot(w_update_list_0, train_loss_list, '-', label='train_loss_list')
# plt.plot(w_update_list_4, train_loss_list, '-', label='train_loss_list')
# plt.plot(w_update_list_9, train_loss_list, '-', label='train_loss_list')
# plt.show()