import os.path
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from r2unet import *
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import torch
from torch.utils.data import DataLoader
# from torch.utils.data import DataLoader
from torch import nn
from datasets import CityscapesDataset
from sklearn.metrics import accuracy_score
from torchvision import transforms
import numpy as np
from scipy import spatial
from sklearn.metrics import f1_score

# our CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default="/src/datasets/cityscapes/", help="directory the Cityscapes dataset is in")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")

args = parser.parse_args()
lr = 0.0002
epochs = 30


# cityscapes dataset loading
tfs = transforms.Compose([transforms.RandomCrop(320),
    transforms.RandomRotation((0,90)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5)])

img_data = CityscapesDataset(args.datadir, split='train', mode='fine', transforms=tfs)
img_batch = torch.utils.data.DataLoader(img_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_dataset = CityscapesDataset(args.datadir, transforms=tfs)
print(img_data)

num_classes = 34 
lo = nn.BCELoss()
##generator as model
generator = R2U_Net().cpu()

gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
train_losses = []
# epoch = 0
##the training loop!!!
def train(init_epoch=0):
    for epoch in range(epochs):
 
      running_loss = 0.0 
  
      for idx_batch, (imagergb, labelmask, labelrgb) in enumerate(img_batch):
            
            gen_optimizer.zero_grad()

            x = imagergb.cpu()
        
            y_ = labelmask.unsqueeze(1).cpu().float()
           
        
            y = generator(x) 
            y = torch.sigmoid(y)      
            loss = lo(y, y_)
            loss.backward()
            gen_optimizer.step()
            running_loss += loss.item()/len(img_batch)
            if idx_batch == 5:
                break
            print('epoch {}, train loss: {:.3f}'.format(epoch, running_loss))
            train_losses.append(running_loss)
            with open("trainresult.csv",'a+', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(["Epoch", "Train_Loss"])
                writer.writerow([epoch, train_losses])

    torch.save(generator.state_dict(),'/src/'+str(epoch)+'generator.pt')              
    m = [j for j in range(5*epochs)]
    plt.xlabel("epochs")  
    plt.ylabel("loss")
    plt.plot(m, train_losses,color = 'b', label = "Training Loss")
    plt.savefig("trainingLoss.png"%())              


val_loader = DataLoader(dataset=val_dataset,
                          batch_size=1,
                          num_workers=0,
                          shuffle=True)

def test():
    generator.eval() 
    
    total = 0
    correct = 0
    running_loss = 0.0
    
    inters = [0 for i in range(19)]
    unions = [0 for i in range(19)]
    for iter, (inputs, labels,labelrgb) in tqdm(enumerate(val_loader)):
        inputs, labels = inputs, labels.long()
        inputs, labels = inputs.cpu(), labels.unsqueeze(1).cpu().float()

        outputs = generator(inputs)
        outputs = torch.sigmoid(outputs)
        loss = lo(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        
        outputs = outputs.cpu()
        outputs = outputs.detach().numpy()
        labels = labels.cpu()
        labels = labels.detach().numpy()
        labels1 = np.ndarray.flatten(labels)

        predict = np.argmax(outputs, axis = 1)
        correct += np.where(predict == labels, 1, 0).sum()
        total += predict.size
        pred = np.ndarray.flatten(predict)

        curr_in, curr_un = iou(predict, labels)
        inters = [inters[p]+curr_in[p] for p in range(len(inters))]
        unions = [unions[p]+curr_un[p] for p in range(len(unions))]
        if iter == 5:
            break

    ious = [inters[p]/unions[p] if unions[p]!=0 else 0 for p in range(len(inters))]
    avg_iou = sum(ious)/len(ious)
    test_acc = 100 * (correct/total)      
    print("##############################Test Accuracy from test function###########################################")
    print('Test accuracy : %.3f' % (test_acc))
    acc = accuracy_score(labels1, pred)              
    f1 = f1_score(labels1,pred, average= "macro")
    print("f1_score",f1)
    print("#################################dice_coef from test function#################################")
    d = dice_coeff(pred, labels1, empty_score=1.0)
    print(dice_coeff(pred, labels1, empty_score=1.0))
    print("#################################Jaccard Index from test funcion#################################")
    j = jaccard_score(labels1,pred, average= "macro")
    print("The jaccard Index is", jaccard_score(labels1,pred, average= "macro"))

    print("#################################accuracy_score from test function#################################")
    print("acc: ", acc)

    with open("testresult.csv",'a+', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["f1_score","dice_coefficient","jaccard_score","accuracy_score"])
        writer.writerow([f1, d,j,acc])




def iou(pred, labels):
    intersections, unions = [], []
    bad_ids = [0,1,2,3,4,5,6,9,10,14,15,16,18, 29,30]
    n_class = 34
    for cls in range(n_class):
        if cls not in bad_ids:
            TP = ((pred==cls) & (labels==cls)).sum()
            FP = ((pred==cls) & (labels!=cls)).sum()
            FN = ((pred!=cls) & (labels==cls)).sum()
            # Complete this function
            intersection = TP
            union = (TP+FP+FN)
            if union == 0:
                intersections.append(0)
                unions.append(0)
                # if there is no ground truth, do not include in evaluation
            else:
                intersections.append(intersection)
                unions.append(union)
                # Append the calculated IoU to the list ious
    return intersections, unions
def dice_coeff(val1, val2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""
   
    val1 = np.asarray(val1).astype(np.bool)
    val2 = np.asarray(val2).astype(np.bool)

    if val1.shape != val2.shape:
        raise ValueError("Shape mismatch error")

    val1 = val1 > 0.5
    val2 = val2 > 0.5

    val_sum = val1.sum() + val2.sum()
    if val_sum == 0:
        return empty_score

    # Computing Dice coefficient here
    intersection = np.logical_and(val1, val2)
    return (2. * intersection.sum() / val_sum)

valid_loss = []
      
def val(epoch):
    generator.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    generator.eval()

    for epoch in range(epochs):
        for j, (inputs, labels, labelrgb) in enumerate(val_loader):
            
            
            inputs = inputs.cpu()
            labels = labels.unsqueeze(1).cpu().float()##groundTruth

            outputs = generator(inputs) ###predictor
            outputs = torch.sigmoid(outputs)
            loss = lo(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            

            if j == 5:
                    break
            valid_loss.append(running_loss)        
                
            outputs = outputs.cpu()
            outputs = outputs.detach().numpy()
            labels = labels.cpu()
            labels = labels.detach().numpy()
            labels1 = np.ndarray.flatten(labels)


                # Calculating the accuracy
            predict = np.argmax(outputs, axis = 1)
            correct = np.where(predict == labels, 1, 0).sum()
            total = predict.size
            val_pix_acc = 100.*correct/total
            pred = np.ndarray.flatten(predict)

            acc = accuracy_score(labels1, pred)              
            f1 = f1_score(labels1,pred, average= "macro")
            print("f1_score",f1)
            print("#################################dice_coef#################################")
            d = dice_coeff(pred, labels1, empty_score=1.0)
            print(dice_coeff(pred, labels1, empty_score=1.0))
            print("#################################Jaccard Index#################################")
            j = jaccard_score(labels1,pred, average= "macro")
            print("The jaccard Index is", jaccard_score(labels1,pred, average= "macro"))
        
            print("#################################accuracy_score#################################")
            print("acc: ", acc)


    with open("valresult.csv",'a+', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["f1_score","dice_coefficient","jaccard_score","accuracy_score"])
        writer.writerow([f1, d,j,acc])

    # torch.save(generator.state_dict(),'/src/'+str(epoch)+'model.pt')              
    m = [j for j in range(5*epochs)]
    plt.xlabel("epochs")  
    plt.ylabel("loss")
    plt.plot(m, valid_loss,color = 'b', label = "Validation Loss")
    plt.savefig("ValidationLoss.png"%())     
    # with open("results.csv",'a+', newline='') as csv_file:
    #         writer = csv.writer(csv_file, delimiter=',')
    #         writer.writerow(["f1_score","dice_coefficient","jaccard_score","accuracy_score"])
    #         writer.writerow([f1, d,j,acc])
    # return f1, dice_coeff(pred, labels1, empty_score=1.0), jaccard_score

# m = [j for j in range(5*epochs)]
# plt.xlabel("epochs")  
# plt.ylabel("loss")
# plt.plot(m, valid_loss,color = 'b', label = "Validation Losses")
# plt.savefig("validationLoss.png"%())

if __name__ == "__main__":
    train()
    val(1) 
    test() 
