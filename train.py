import time
import os
import sys
from tqdm import tqdm

import torch
from dataloader.data_loaders import LaneDataSet
from dataloader.transformers import Rescale
from model.model import LaneNet, compute_loss
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms

from utils.cli_helper import parse_args
from utils.average_meter import AverageMeter

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform=transforms.Compose([Rescale((512, 256))])

def train(train_loader, model, optimizer, epoch,max_n_objects):
            
    '''
    train_loader :Training data package collection
    model : Specific model
    optimizer : optimization method
    epoch : Number of iterations
    '''
    # hyperparameter c
    c = 1.005
    # Parameter initialization
    batch_time = AverageMeter()
    mean_iou = AverageMeter()
    mean_p = AverageMeter()
    mean_r = AverageMeter()
    mean_acc = AverageMeter()
    total_losses = AverageMeter()
    binary_losses = AverageMeter()
    instance_losses = AverageMeter()
    end = time.time()
    step = 0    
    # Unpack data
    t = tqdm(enumerate(iter(train_loader)), leave=False, total=len(train_loader))
    for batch_idx, batch in t:
        step += 1
        # Data label initialization
        image_data = Variable(batch[0]).type(torch.FloatTensor).to(DEVICE)
        binary_label = Variable(batch[1]).type(torch.LongTensor).to(DEVICE)
        instance_label = Variable(batch[2]).type(torch.FloatTensor).to(DEVICE)
        # Forward calculation
        net_output = model(image_data)
        # Calculate classification weight
        n,l,w = binary_label.size()
        p = torch.sum(binary_label)/(n*l*w)
        p_class = p*binary_label+(1-p)*(1-binary_label).to(DEVICE)
        class_weights = 1/(torch.log(c+p_class))
        # Calculation type
        zeros = torch.zeros((l,w)).to(DEVICE)
        n_objects = torch.zeros(n).int()
        num = -1
        for label in instance_label:
            count = 0
            num += 1
            for label_i in range(label.size(0)):
                if False not in (label[label_i] == zeros):
                    n_objects[num] = count
                    break
                else:
                    count += 1
        
        # Calculation loss function
        out = compute_loss(net_output, binary_label,instance_label,n_objects,max_n_objects,class_weights)
        total_loss, binary_loss, instance_loss, train_bin_iou,train_bin_p,train_bin_r,train_bin_acc = out
        # Update each loss function item
        total_losses.update(total_loss.item(), image_data.size()[0])
        binary_losses.update(binary_loss.item(), image_data.size()[0])
        instance_losses.update(instance_loss.item(), image_data.size()[0])
        mean_iou.update(train_bin_iou, image_data.size()[0])
        mean_p.update(train_bin_p, image_data.size()[0])
        mean_r.update(train_bin_r, image_data.size()[0])
        mean_acc.update(train_bin_acc, image_data.size()[0])
        # Update weight
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # Update calculation time point
        batch_time.update(time.time()-end)
        end = time.time()

        # Output the results every 10 batches
        if step % 10 == 0:
            print(
                "Epoch {ep} Step {st} |({batch}/{size})| ETA: {et:.4f}|Total loss:{tot:.5f}|Binary loss:{bin:.5f}|Instance loss:{ins:.5f}|IoU:{iou:.5f}|P:{p:.5f}|R:{r:.5f}|ACC:{acc:.5f}".format(
                    ep=epoch + 1,
                    st=step,
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    et=batch_time.val,
                    tot=total_losses.avg,
                    bin=binary_losses.avg,
                    ins=instance_losses.avg,
                    iou=train_bin_iou,
                    p = train_bin_p,
                    r = train_bin_r,
                    acc = train_bin_acc
                ))
        # Cache refresh
        sys.stdout.flush()

# Save model
def save_model(save_path, epoch, model):
    save_name = os.path.join(save_path, f'{epoch+1}_checkpoint.pth')
    torch.save(model, save_name)
    print("model is saved: {}".format(save_name))


def main(type = 'formal_train'):
    # Important parameter initialization
    args = parse_args()
    # Storage path settings
    save_path = args.save
    # Detect whether the folder exists, and create a new folder if it does not exist
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # Create training set path
    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    # Maximum number of species
    if type == 'pre_train':
        max_n_objects = 5
    else:
        max_n_objects = 3
    # Create training set data
    train_dataset = LaneDataSet(train_dataset_file,n_labels=max_n_objects,transform=transforms.Compose([Rescale((512, 256))]),shuffle=False)
    # Load dataset and package
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False,num_workers=4,pin_memory=True)
    # Initialization model
    # pre_train
    if type == 'pre_train':
        model = LaneNet()
        model.to(DEVICE)
    # formal train
    else:
        model_path = "Your pre training model"
        model = torch.load(model_path,map_location = DEVICE)
    # Select Adam as the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Displays the total number of iterations and the total amount of training data
    print(f"{args.epochs} epochs {len(train_dataset)} training samples\n")
    # Displays the current number of iterations
    for epoch in range(0, args.epochs):
        print(f"Epoch {epoch}")
        # Training model
        train(train_loader, model, optimizer, epoch,max_n_objects)
        # Save the model every 100 training times
        if (epoch + 1) % 100 == 0:
            save_model(save_path, epoch, model)

if __name__ == '__main__':
    main(type = 'pre_train')
