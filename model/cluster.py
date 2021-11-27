from sklearn.cluster import KMeans,MiniBatchKMeans
import torch
def kmeans(lane_data_embedding,n_clusters,type='kmeans'):
    '''
        lane_data: The embedding of the input road is preferably numpy type, and the data is located in the CPU
        n_cluster: Cluster type
        type: The parameters of clustering method,such as "kmeans","kemans++","minibatchmeans"
    '''
    if(type == 'kmeans'):
        cluster_result = KMeans(n_clusters,init='random').fit(lane_data_embedding)
    elif(type  == 'kmeans++'):
        cluster_result = KMeans(n_clusters).fit(lane_data_embedding)
    elif(type =='minibatchmeans'):
        cluster_result = MiniBatchKMeans(n_clusters).fit(lane_data_embedding)
    else:
        print("type has a error")
    result_indexs = [0,1]
    kmeans_result = torch.from_numpy(cluster_result.labels_)
    return kmeans_result,result_indexs
    
def order_list(kmeans_center):
    result = []
    order = sorted(kmeans_center)
    for data in kmeans_center:
        result.append(order.index(data))
    return result
