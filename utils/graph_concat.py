'''
@Project: HACH-MVC
@Author : Ali Takrar
@Email  : takrar.co@gmail.com
@File   : graph_concat
@Desc
    
'''
import torch

def all_fg_consstruct(features, adj_new,true_viewnum):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    view_graph = torch.eye(features[0].shape[0]).to(device)
    view_graph = torch.repeat_interleave(view_graph,repeats = true_viewnum, dim = 1)
    view_graph = torch.repeat_interleave(view_graph, repeats=true_viewnum, dim=0)

    for i in range(true_viewnum):
        if i == 0:
            adj_graph = adj_new[0]
        else:
            adj_graph = adjConcat(adj_graph, adj_new[i])
    # adj_new = torch.add(adj_graph , view_graph)
    # adj_new = torch.sub(adj_new-torch.eye(features[0][0].shape[0]*features.shape[1]))
    adj_new = adj_graph + view_graph
    adj_new = adj_new - torch.eye(features[0].shape[0] * true_viewnum).to(device)
    return adj_new
    


def adjConcat(a, b):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Merge the two matrices a and b diagonally, and fill the empty spaces with zeros [a, 0.0, b]
    # Get the dimensions of a and b. First, merge the zero matrices of a and b*a by row (vertically) to get c, and then merge the zero matrices of a*b and b by row to get d
    # Merge c and d horizontally
    #    '''
    lena = a.shape[0]  # len(a)
    lenb = b.shape[0]  # len(b)
    p = torch.zeros((lenb, lena)).to(device)
    q = torch.zeros((lena, lenb)).to(device)
    left = torch.vstack((a.to_dense(), p))  # First vertically concatenate a and a zero matrix of len(b)*len(a) to get the left half
    right = torch.vstack((q, b.to_dense()))  # Then vertically concatenate a zero matrix of len(a)*len(b) with b to get the right half
    result = torch.hstack((left, right))  # Concatenate the left and right matrices horizontally
    return result