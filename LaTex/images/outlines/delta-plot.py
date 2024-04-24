from matplotlib import pyplot as plt
import torch

COLOR_0 = "#5735C3"
COLOR_1 = "#ADE216"
COLOR_2 = "#B028D0"
COLOR_3 = "#C37935"
COLOR_4 = "#578EA1"
COLOR_5 = "#E81086"
COLOR_6 = "#FAA6FC"

def plotDelta():
    fig = plt.figure(dpi=96,figsize=(9, 5.3))
    plt.subplots_adjust(wspace=0.3)
    ax = fig.add_subplot(1,1,1)
    
    x_tensor = torch.tensor([-5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3, -5.2, -5.1, -5.0, 
                            -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3, -4.2, -4.1, -4.0,
                            -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0,
                            -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0,
                            -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0,
                            -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
                            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                            2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                            3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                            4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
                            5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9])
    y_tensor = torch.sigmoid(x_tensor)
    x = list(x_tensor)
    y = list(y_tensor)
    
    SIZE_LOSS = 6
    ax.errorbar(x, y, ms=SIZE_LOSS - 2, color=COLOR_0, label=r'Sigmoid')

    y_tensor = torch.tanh(x_tensor)
    x = list(x_tensor)
    y = list(y_tensor)
    ax.errorbar(x, y, ms=SIZE_LOSS - 2, color=COLOR_1, label=r'Tanh')

    y_tensor = torch.relu(x_tensor)
    x = list(x_tensor)
    y = list(y_tensor)
    ax.errorbar(x, y, ms=SIZE_LOSS - 2, color=COLOR_2, label=r'ReLU')

    x_tensor_ = torch.tensor([-5.5,  -5.0, -4.5,  -4.0, 
                             -3.5, -3.0, -2.5, -2.0, -1.5, -1.0,
                            -0.5,  0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                            3.5, 4.0, 4.5, 5.0, 5.5])
    y_tensor = torch.softmax(x_tensor_, dim=0)
    x = list(x_tensor_)
    y = list(y_tensor)
    ax.errorbar(x, y, ms=SIZE_LOSS - 2, color=COLOR_3, label=r'Softmax')

    def swish(x):
        return x * torch.sigmoid(x)
    y_tensor = swish(x_tensor)
    x = list(x_tensor)
    y = list(y_tensor)
    ax.errorbar(x, y, ms=SIZE_LOSS - 2, color=COLOR_4, label=r'Swish')


    # 将坐标图上的上边框和右边框隐藏
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 将坐标图上的下边框和左边框作为x轴和y轴，并调整位置
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('axes', 0.5))

    ax.set_xlabel('x', loc='right')
    ax.set_ylabel('y', loc='top')
    ax.set_xlim((-6, 6))
    ax.set_ylim((-1.2, 1.2))

    ax.legend(loc="best")
    # ax.set_title('(b) '+ "Training loss $vs.$ No. of epoch", y=-0.268, fontsize=16)

    plt.show()

# Main
if __name__ == "__main__":
    plotDelta()
