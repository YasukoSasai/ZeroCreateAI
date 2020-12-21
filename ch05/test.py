# ========= 今回の発表で行った学習 =========(5.7.4)
import sys
sys.path.append('../')
from dataset.mnist import load_mnist
from two_layers_net import TwoLayerNet
from PIL import Image #画像表示にはPILモジュールを使う。
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ========= データを取得(入力データを一次元化、正解ラベルをone_hot_label化) ========= 
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(" --------- データ内容 --------- ")
print("x_train.shape", x_train.shape) #訓練データの入力データ
print("t_train.shape", t_train.shape) #訓練データの正解データ 
print("x_test.shape", x_test.shape) #テストデータの入力データ
print("t_test.shape", t_test.shape) #テストデータの正解データ

# ========= ディープラーニングのモデル ========= 
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10) #入力層-中間層-出力層

# ========= ハイパーパラメータ ========= 
iters_num = 10000 #学習回数(パラメータ更新回数) iters = iterarion
train_size = x_train.shape[0] #x_trainデータ形状の0次元の数 = 60000枚
batch_size = 100 #batch_size = １回の学習で何枚分まとめて行うか
learning_rate = 0.1 #学習率

train_loss_list = [] #学習ごとの誤差(損失関数の結果)を格納するためのリスト
train_acc_list = [] #訓練データ(学習)における正確率を格納するためのリスト
test_acc_list = [] #テストデータ(テスト)における正確率を格納するためのリスト

iter_per_epoch = max(train_size / batch_size, 1) #1エポックあたりの学習回数。600回。

# ========= 学習フェーズ ========= 
print(" ========= 学習開始 ========= ")
started_time = datetime.now()

for i in range (iters_num): #10000回繰り返し
    # ========= ミニバッチの取得 =========
    batch_mask = np.random.choice(train_size, batch_size) #60000個の中から100個ランダムでインデックスを取り出す
    x_batch = x_train[batch_mask] #100個の入力画像
    t_batch = t_train[batch_mask] #100個の入力画像に対する正解データ

    if i == 1:
        print(" --------- １回の学習で使われるバッチデータ --------- ")
        print("x_batch.shape", x_batch.shape)
        print(str(x_batch.shape[0]) + "枚分、"+ str(x_batch.shape[1]) + "ピクセル/枚")
        print("t_batch.shape", t_batch.shape)
        print(str(t_batch.shape[0]) + "枚分の"+ str(t_batch.shape[1]) + "個の正解ラベル")

    # ========= 勾配計算 ========= 
    #数値微分　時間かかる
    # grad = network.numerical_gradient(x_batch, t_batch)
    #誤差逆伝播法　高速！ 
    grad = network.gradient(x_batch, t_batch)

    # ========= パラメータ更新 ========= 
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

finished_time = datetime.now()
print("学習時間", finished_time - started_time)

# ========= テストフェーズ ========= 
print(" ========= テスト開始 ========= ")
started_time2 = datetime.now()
# ========= 複数枚でテストしたい時(画像表示の仕方も変更する) ========= 
# test_size = x_test.shape[0]
# batch_mask = np.random.choice(test_size, 10) #何枚でテストを行いたいか
# print("batch_mask",batch_mask)
# #train_size枚の中からbatch_size枚ランダムで配列で取り出す
# x_test = x_test[batch_mask] #100個の入力画像
# t_test = t_test[batch_mask] #100個の入力画像に対する正解データ
# test_acc = network.accuracy(x_test, t_test)
# test_acc_list.append(test_acc)
# print(test_acc_list)

# ========= １枚でテストしたい時 ========= 
x_test = x_test[[500]] 
t_test = t_test[[500]] 
test_acc = network.accuracy(x_test, t_test)

finished_time2 = datetime.now()
print("テスト時間", finished_time2 - started_time2)

# ========= テストの精度 ========= 
print("test acc: "+ str(test_acc))


# # ========= テストデータの画像表示 ========= 
# def ConvertToImg(img):
#     return Image.fromarray(np.uint8(img))

# chr_w = 28 # MNIST一文字の幅
# chr_h = 28 # MNIST一文字の高さ
# num = 20 # 表示する文字数

# # MNISTの文字をPILで１枚の画像に描画する
# canvas = Image.new('RGB', (int(chr_w * num/2), int(chr_h * num/2)), (255, 255, 255))

# i = 0

# #画像表示のためにunnormalizedデータ読み込み(もっといい方法あるかな‥) 
# (x_train, t_train), (x_test, t_test) = \
#     load_mnist(flatten=True, normalize=False)
# # ========= １枚テストの際の画層表示 ========= 
# chrImg = ConvertToImg(x_test[4].reshape(chr_w, chr_h))
# canvas.paste(chrImg, (chr_w*i, chr_h))
# canvas.show()
# # ========= 複数テストの際の画像表示 ========= 
# for item in batch_mask:
#     chrImg = ConvertToImg(x_test[item].reshape(chr_w, chr_h))
#     canvas.paste(chrImg, (chr_w*i, chr_h))
#     i = i + 1

# canvas.show()

