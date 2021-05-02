import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

epoch = range(1,101)
print(epoch)
hidden_units_list = [32, 64, 128, 256, 512]
num_batch_list = [32, 64, 128, 256, 512]
learning_rate_list = [0.005, 0.001, 0.0005, 0.0001]


# hidden units
df = pd.read_csv('accuracy_hidden_units.csv', header=None)
accuracy = df.to_numpy()
print(np.shape(accuracy))
print(accuracy)

for i in range(len(accuracy)):
    plt.plot(epoch, accuracy[i], label="Hidden units: %s" % hidden_units_list[i])
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
#plt.xticks(epoch)
plt.ylim(60,90)
plt.legend()
plt.grid()
plt.title('Test accuracies with different number of hidden units')
plt.savefig('diff hidden units.jpg')
plt.show()

# learning rate
df = pd.read_csv('accuracy_learning_rate.csv', header=None)
accuracy = df.to_numpy()
print(np.shape(accuracy))
print(accuracy[0])

for i in range(len(accuracy)):
    plt.plot(epoch, accuracy[i], label="Learning_rate: %s" % learning_rate_list[i])
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
#plt.xticks(epoch)
#plt.ylim(0,100)
plt.legend()
plt.grid()
plt.title('Test accuracies with different learning rate')
plt.savefig('diff learning rate.jpg')
plt.show()

# batch size
df = pd.read_csv('accuracy_batch_size.csv', header=None)
accuracy = df.to_numpy()
print(np.shape(accuracy))


for i in range(len(accuracy)):
    plt.plot(epoch, accuracy[i], label="Batch size: %s" % num_batch_list[i])
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
#plt.xticks(epoch)
#plt.ylim(60,90)
plt.legend()
plt.grid()
plt.title('Test accuracies with different number of batch size')
plt.savefig('diff batch size.jpg')
plt.show()

# batch size loss
df = pd.read_csv('accuracy_batch_size.csv', header=None)
loss = df.to_numpy()
print(np.shape(loss))

'''
for i in range(len(loss)):
    plt.plot(epoch, loss[i], label="Batch size: %s" % num_batch_list[i])
plt.xlabel("Epoch")
plt.ylabel("Avg. Loss")
#plt.xticks(epoch)
#plt.ylim(60,90)
plt.legend()
plt.grid()
plt.title('Avg. train loss with different number of batch size')
plt.savefig('loss with diff batch size.jpg')
plt.show()
'''