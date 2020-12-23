#========= どのように画像表示するか ==========(3,6,1)
import sys
sys.path.append('../')
from dataset.mnist import load_mnist 
import numpy as np

(x_train, t_train), (x_test, t_test) = \
  load_mnist(flatten=True, normalize=False)
"""
  flatten:入力画像を一次元にするかどうか、TrueなのでNumpy配列に1次元(784)で格納。→なので後ほど2次元(28,28)に再変形。
  normalize:入力画像を０〜１の値に正規化するかどうか。
  （one_hot_label:ラベルをone_hot表現で格納するかどうか。one_hot表現＝正解ラベル１，それ以外０）
"""
# ========= MNIST画像を表示する =========
from PIL import Image #画像表示にはPILモジュールを使う。

def img_show(img):  
  pil_img = Image.fromarray(np.uint8(img)) 
  pil_img.show()
img = x_test[800]
print(img[0])

# ========= 画像のデータを(28*28)で出す =========
(x_train, t_train), (x_test, t_test) = \
  load_mnist(flatten=True, normalize=True)

normalized_img = x_test[800]
print(normalized_img[0])
k = 0
aaa = []
for i in range(28):
  for j in range(28):
    aaa.append(normalized_img[k])
    k += 1
  print(aaa)
  aaa = []

img = img.reshape(28, 28) 
img_show(img)