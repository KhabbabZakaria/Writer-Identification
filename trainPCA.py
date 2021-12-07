CUDA_LAUNCH_BLOCKING=1

from dgmp import *
from model import *
import torch
import torch.nn.functional as F
from data import *
from loss import ContrastiveLoss
from paths import *
import matplotlib.pyplot as plt
from crow import *
from sklearn import decomposition
import time
# starting time
start = time.time()

#go to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

resnet = resnet18().cuda()


pca = decomposition.PCA()
pca.n_components = 100

class Model(torch.nn.Module):
    def __init__(self, model, pool):
        super(Model, self).__init__()
        self.model = model
        self.pool = pool

    def forward(self, sample):
        output = self.model(sample)

        ####################
        ### Aggregation ###
        ####################
        output = self.pool(output) #dgmp or GeM
        #output =  torch.sum(output.view(output.size(0), output.size(1), -1), dim=2) #SPoC
        #alist = [] #crow
        #for i in range(output.shape[0]): #crow
        #    output2 = apply_crow_aggregation(output[i]) #crow
        #    alist.append(output2) #crow
        #output = torch.stack(alist) #crow

        #########################
        ### Dimension Process ###
        #########################
        output = F.normalize(output, p=2, dim=1) #L2 Normalizartion
        return output

model = Model(model=resnet, pool=dgmp)
model.eval()

criterion = ContrastiveLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) #lr = 1e-4 for others than crow

print('Importing Model, Loss, Optimizer Done')


print('Training started')

counter = []
counter_test = []
loss_history = []
loss_history_test = []
iteration_number= 0
iteration_number_test= 0

for j in range(epochs):
    #train
    for e, (inputs1, inputs2, labels) in enumerate(train_dataloader):
      opt.zero_grad()
      trainlist1 = []
      trainlist2 = []
      for i in range(0, batch_size, minibatch_size):
        output1 = model.forward(inputs1[i:i+minibatch_size].cuda())
        output2 = model.forward(inputs2[i:i+minibatch_size].cuda())
        trainlist1.append(output1)
        trainlist2.append(output2)
      output1 = torch.cat(trainlist1)
      output2 = torch.cat(trainlist2)
      output1 = torch.tensor(pca.fit_transform(output1.cpu().detach().numpy()), requires_grad=True).cuda()  #PCA
      output2 = torch.tensor(pca.fit_transform(output2.cpu().detach().numpy()), requires_grad=True).cuda()  #PCA
      loss_contrastive = criterion(output1, output2, labels.cuda())
      loss_contrastive.backward()
      opt.step()
      if e % 10 == 0:
          print("Epoch number {}\n Current loss {}\n".format(j, loss_contrastive.item()))
          iteration_number += 10
          counter.append(iteration_number)
          loss_history.append(loss_contrastive.item())
    #test
    for e, (inputs1, inputs2, labels) in enumerate(test_dataloader):
      testlist1 = []
      testlist2 = []
      for i in range(0, batch_size, minibatch_size):
        output1 = model.forward(inputs1[i:i+minibatch_size].cuda())
        output2 = model.forward(inputs2[i:i+minibatch_size].cuda())
        testlist1.append(output1)
        testlist2.append(output2)
      output1 = torch.cat(trainlist1)
      output2 = torch.cat(trainlist2)
      output1 = torch.tensor(pca.fit_transform(output1.cpu().detach().numpy()), requires_grad=False).cuda()  #PCA
      output2 = torch.tensor(pca.fit_transform(output2.cpu().detach().numpy()), requires_grad=False).cuda()  #PCA
      loss_contrastive = criterion(output1, output2, labels.cuda())
      if e % 10 == 0:
          iteration_number_test += 10
          counter_test.append(iteration_number_test)
          loss_history_test.append(loss_contrastive.item())
plt.plot(counter, loss_history)
plt.plot(counter_test, loss_history_test)
plt.legend(["Train", "Test"])
plt.savefig("Training and Evaluation.png")
plt.show()

print('Training finished')

#torch.save(model, PATH)
torch.save(model.state_dict(), PATH)

print('Model Saved...')

# end time
end = time.time()

# total time taken
print("Runtime of the program is", end - start)


