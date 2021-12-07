from dgmp import *
from model import *
import torch
import torch.nn.functional as F
from data import *
from loss import ContrastiveLoss
from paths import *
import matplotlib.pyplot as plt
from crow import *
import time

# starting time
start = time.time()

#go to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

resnet = resnet18().cuda()




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
        #output = self.pool(output) #dgmp or GeM
        output =  torch.sum(output.view(output.size(0), output.size(1), -1), dim=2) #SPoC
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

model = Model(model=resnet, pool=gmp)
#model.eval()

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
    for e, (inputs1, inputs2, labels) in enumerate(train_dataloader):
      opt.zero_grad()
      output1 = model.forward(inputs1.cuda())
      output2 = model.forward(inputs2.cuda())
      loss_contrastive = criterion(output1, output2, labels.cuda())
      loss_contrastive.backward()
      opt.step()
      if e % 10 == 0:
          print("Epoch number {}\n Current loss {}\n".format(j, loss_contrastive.item()))
          iteration_number += 10
          counter.append(iteration_number)
          loss_history.append(loss_contrastive.item())

    for e, (inputs1, inputs2, labels) in enumerate(test_dataloader):
      output1 = model.forward(inputs1.cuda())
      output2 = model.forward(inputs2.cuda())
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


