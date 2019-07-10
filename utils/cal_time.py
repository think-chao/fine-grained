import time
import matplotlib.pyplot as plt


def tic():
    globals()['tt'] = time.clock()


def toc():
    print('\nIteration time: %.3f seconds\n' % (time.clock() - globals()['tt']))


def vis(img, target, predict):
    plt.imshow(img)
    plt.title('True:{} Pred:{}'.format(target, predict))
    plt.show()
