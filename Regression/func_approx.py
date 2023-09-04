'''
#PaddldPaddle for function approximation
#Author: Xuhui Meng(xuhui_meng@hust.edu.cn)
'''
import paddle as pd
import paddle.nn.functional as F
pd.device.set_device('cpu')
#pd.device.set_device('gpu:0')
import numpy as np
import time
import matplotlib.pyplot as plt

pd.set_default_dtype('float32')
pd.seed(1234)

def main():
    x = np.linspace(-1., 1., 21).astype(np.float32)
    x_train = x.reshape((-1, 1))
    y_train = np.sin(5*x_train)
    x_train = pd.to_tensor(x_train)
    y_train = pd.to_tensor(y_train)

    model = pd.nn.Sequential(
            pd.nn.Linear(1, 20),
            pd.nn.Tanh(),
            pd.nn.Linear(20, 20),
            pd.nn.Tanh(),
            pd.nn.Linear(20, 20),
            pd.nn.Tanh(),
            pd.nn.Linear(20, 1)
            )


    opt = pd.optimizer.Adam(learning_rate=1.0e-3, parameters=model.parameters())

    model.train()
    
    nmax = 10000
    n = 0
    while n <= nmax:
        n += 1
        y_pred = model(x_train)
        loss = F.mse_loss(y_pred, label=y_train)
        loss.backward()
        opt.step()
        opt.clear_grad()

        if n%100 == 0:
            print('steps: %d, loss: %.3e'%(n, float(loss)))

    num_test = 101
    x_test = np.linspace(-1., 1., num_test).astype(np.float32)
    x_test = x_test.reshape((-1, 1))
    y_ref = np.sin(5*x_test)
    x_test = pd.to_tensor(x_test)
    y_test = model(x_test)

    plt.figure()
    plt.plot(x_test.numpy(), y_ref, 'k-')
    plt.plot(x_train.numpy(), y_train.numpy(), 'bo')
    plt.plot(x_test.numpy(), y_test.numpy(), 'r--')
    plt.show()
    
        
    

if __name__ == '__main__':
    main()
