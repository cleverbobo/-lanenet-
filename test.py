import time
import os
from tqdm import tqdm

import torch
from dataloader.data_loaders import LaneDataSet
from dataloader.transformers import Rescale
from model.model import compute_loss
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms

from utils.cli_helper import parse_args
from utils.average_meter import AverageMeter

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform=transforms.Compose([Rescale((512, 256))])

def test(test_loader, model):
    '''
    test_loader :Training data package collection
    model : Specific model
    optimizer : optimization method
    epoch : Number of iterations
    '''
    model.eval()
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
    # Unpack data
    t = tqdm(enumerate(iter(test_loader)), leave=False, total=len(test_loader))
    # Maximum number of species
    max_n_objects = 3
    for batch_idx, batch in t:
        # Data label initialization
        image_data = Variable(batch[0]).type(torch.FloatTensor).to(DEVICE)
        binary_label = Variable(batch[1]).type(torch.LongTensor).to(DEVICE)
        instance_label = Variable(batch[2]).type(torch.FloatTensor).to(DEVICE)
        end = time.time()
        # Forward calculation
        with torch.no_grad():
            net_output = model(image_data)
        batch_time.update(time.time() - end)
        end = time.time()
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
        total_loss, binary_loss, instance_loss, test_bin_iou,test_bin_p,test_bin_r,test_bin_acc = out
        # Update each loss function item
        total_losses.update(total_loss.item(), image_data.size()[0])
        binary_losses.update(binary_loss.item(), image_data.size()[0])
        instance_losses.update(instance_loss.item(), image_data.size()[0])
        mean_iou.update(test_bin_iou, image_data.size()[0])
        mean_p.update(test_bin_p, image_data.size()[0])
        mean_r.update(test_bin_r, image_data.size()[0])
        mean_acc.update(test_bin_acc, image_data.size()[0])      
        # Output test data
    print(f"IOU:{ mean_iou.avg} P:{ mean_p.avg} R:{mean_r.avg} ACC:{ mean_acc.avg} time:{batch_time.avg}")



def main():

    # Important parameter initialization
    args = parse_args()
    # Create training set path
    test_dataset_file = os.path.join(args.dataset, 'The file that contains the test set path')
    # Create training set data
    test_dataset = LaneDataSet(test_dataset_file,transform=transforms.Compose([Rescale((512, 256))]),shuffle=False)
    # Load dataset and package
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,num_workers=4,pin_memory=True)
    # Initialization model
    model_path = "your model file"
    model = torch.load(model_path,map_location = DEVICE)
    # test data
    test(test_loader, model)


if __name__ == '__main__':
    main()
