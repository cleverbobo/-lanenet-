from dataloader.transformers import Rescale
from torchvision import transforms
import cv2
import torch
import matplotlib.pyplot as plt
from model.cluster import kmeans
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import time
# Read a single picture and resize
def read_single_image(img_dir,size):
    # resize
    transform = transforms.Compose([Rescale(size)])
    # read picture
    image_cv = transform(cv2.imread(img_dir))
    # Read the picture and convert it into  torch.tensor
    image = torch.from_numpy(image_cv).reshape(3,size[1],size[0])
    # Add dimension to data
    image_torch = (image.unsqueeze(0)).type(torch.FloatTensor).to(DEVICE)
    # Returns the processed picture(size:1,3,256,512)
    return image_torch,image_cv

# Read multiple pictures and resize
def read_multi_image(img_dir_list,size):
    '''
        img_dir_list : List of stored pictures
        size :The size to be changed is consistent with the input of the network. 512 * 256 is used this time
    '''
    # resize
    transform = transforms.Compose([Rescale(size)])
    n = len(img_dir_list)
    image_cv_list = []
    # Load data
    for i in range(n):
        image_cv = transform(cv2.imread(img_dir_list[i]))
        image_cv_list.append(image_cv)
        image = torch.from_numpy(image_cv).reshape(3,size[1],size[0])
        if i != 0:
            image_data = torch.cat((image_data,(image.unsqueeze(0)).type(torch.FloatTensor).to(DEVICE)),dim=0)
        else :
            image_data = (image.unsqueeze(0)).type(torch.FloatTensor).to(DEVICE)
    # Returns the processed picture(size:n,3,256,512)
    return image_data,image_cv_list


# Output the predicted results in the form of pictures
def output_line_image(image_cv_list,predict_result,width,bitch_size,n_clusters):
    '''
        image_cv_list : List of picture data read in CV format
        predict_result: Results of network prediction
        width: width of the picture of the network to calculate the coordinate value
        
    '''
    _,ax=plt.subplots(bitch_size,1)
    for i in range(bitch_size):
        # Screening out lane line area
        area = predict_result[i][0,:]>0
        # Record coordinate values
        Index = torch.linspace(0,len(area)-1,len(area))[area]
        coor_x = Index//width 
        coor_y = Index%width
        lane_data  = predict_result[i][1:,area].cpu().t()
        end = time.time()
        print(n_clusters[i])
        kmeans_result,result_indexs = kmeans(lane_data.detach().numpy(),n_clusters=n_clusters[i]) 
        print(f"Clustering time ：{time.time()-end}")
        cluster_result = torch.cat((lane_data.t(),kmeans_result.view(1,-1),coor_x.view(1,-1),coor_y.view(1,-1)),dim=0)
        # Convert classification results to picture output
        image = image_cv_list[i]
        for num,j in enumerate(cluster_result[4,:]):
            if j==result_indexs[0]:
                image[int(cluster_result[5][num])][int(cluster_result[6][num])][0] = 255
            elif j==result_indexs[1]:
                image[int(cluster_result[5][num])][int(cluster_result[6][num])][1] = 255
            elif j==result_indexs[2]:
                image[int(cluster_result[5][num])][int(cluster_result[6][num])][2] = 255
            elif j==result_indexs[3]:
                image[int(cluster_result[5][num])][int(cluster_result[6][num])][2] = 0
                image[int(cluster_result[5][num])][int(cluster_result[6][num])][1] = 0
                image[int(cluster_result[5][num])][int(cluster_result[6][num])][0] = 0
            elif j==result_indexs[4]:
                image[int(cluster_result[5][num])][int(cluster_result[6][num])][2] = 255
                image[int(cluster_result[5][num])][int(cluster_result[6][num])][1] = 255
                image[int(cluster_result[5][num])][int(cluster_result[6][num])][0] = 0
            else:
                image[int(cluster_result[5][num])][int(cluster_result[6][num])][2] = 255
                image[int(cluster_result[5][num])][int(cluster_result[6][num])][1] = 255
                image[int(cluster_result[5][num])][int(cluster_result[6][num])][0] = 255
        ax[i].imshow(image)
    plt.show()
