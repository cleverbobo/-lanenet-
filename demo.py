import torch
from utils.image_operation import read_multi_image
import cv2
import numpy as np
from model.cluster import kmeans
# fitting curve
from scipy.optimize import least_squares
'''
Image batch clustering, fitting curve output
'''
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def solve_func(k,x,y):
    result = []
    for i in range(len(x)):
        result.append(x[i][0]*k[0]+x[i][1]*k[1]+x[i][2]*k[2]-y[i])
    result = np.asarray(result)
    return result
def print_lane():
    model_path = "./demo/demo.pth"
    img_dir = ("./demo/demo1.jpg","./demo/demo2.jpg","./demo/demo3.jpg","./demo/demo4.jpg")
    # Loading model
    model = torch.load(model_path,map_location = DEVICE)
    model.eval()
    n_clusters = torch.tensor([2,2,2,2])
    size = (512,256)
    # Load data
    image_data,image_cv_list = read_multi_image(img_dir,size)
    # Output results
    with torch.no_grad():
        net_output = model(image_data)
    # The output results are combined and then clustered to distinguish each road label
    result = torch.cat((net_output['binary_seg_pred'],
                        net_output['instance_seg_logits']),dim =1)
    result_view = result.view(4,5,-1)
    width,bitch_size = size[0],4
    for i in range(bitch_size):
        # Screening out lane line area
        area = result_view[i][0,:]>0
        # Record coordinate values
        Index = torch.linspace(0,len(area)-1,len(area))[area]
        coor_x = Index//width 
        coor_y = Index%width
        lane_data  = result_view[i][1:,area].cpu().t()
        kmeans_result,result_indexs = kmeans(lane_data.detach().numpy(),n_clusters=n_clusters[i],type='kmeans') 
        cluster_result = torch.cat((lane_data.t(),kmeans_result.view(1,-1),coor_x.view(1,-1),coor_y.view(1,-1)),dim=0)
        #  Convert classification results to picture output
        p1_x,p1_y,p2_x,p2_y = [],[],[],[]
        image = image_cv_list[i]
        for num,j in enumerate(cluster_result[4,:]):
            if j==result_indexs[0]:
                p1_x.append([int(cluster_result[5][num])**2,int(cluster_result[5][num]),1])
                p1_y.append(int(cluster_result[6][num]))
            else :
                p2_x.append([int(cluster_result[5][num])**2,int(cluster_result[5][num]),1])
                p2_y.append(int(cluster_result[6][num]))

        A1 = least_squares(solve_func,(0,1,1),args=(p1_x,p1_y),bounds=((-0.0001,-np.inf, -np.inf),(0.0001, np.inf,np.inf))).x
        a1_x = np.arange(np.min(np.array(p1_x)[:,1]),256)
        a1_y = np.dot(np.stack((a1_x**2,a1_x,np.ones(a1_x.size)),axis=1),A1.reshape((3,1)))

        A2 = least_squares(solve_func,(0,1,1),args=(p2_x,p2_y),bounds=((-0.0001,-np.inf, -np.inf),(0.0001, np.inf,np.inf))).x
        a2_x = np.arange(np.min(np.array(p2_x)[:,1]),256)
        a2_y = np.dot(np.stack((a2_x**2,a2_x,np.ones(a2_x.size)),axis=1),A2.reshape((3,1)))
        # Set color
        if np.mean(a2_y)>np.mean(a1_y):
            color1 = (0,0,255)
            color2 = (0,255,0)
        else:
            color1 = (0,255,0)
            color2 = (0,0,255)
        for m,(x,y) in enumerate(zip(a1_x,a1_y)):
            if y<=0:
                #If the drawing line exceeds the length of the picture, it will not be drawn
                continue
            if m+1>=a1_x.size:
                break
            cv2.line(image, (y,x),(a1_y[m+1],a1_x[m+1]),color1,thickness=2)
        for m,(x,y) in enumerate(zip(a2_x,a2_y)):
            if y<=0:
                #If the drawing line exceeds the length of the picture, it will not be drawn
                continue
            if m+1>=a2_x.size:
                break
            cv2.line(image, (y,x),(a2_y[m+1],a2_x[m+1]),color2,thickness=2)
        cv2.imshow('img',image)
        cv2.waitKey(0)


if __name__ =="__main__":
    print_lane()
    