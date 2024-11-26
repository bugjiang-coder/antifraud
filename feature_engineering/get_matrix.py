import pickle

import os
import torch
import numpy as np

def create_adjacency_matrix(adj_list, n):

    adj_matrix = np.zeros((n, n))
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1
    return adj_matrix

def block_matrix_multiply(A, B, block_size, device):
    # print('block_matrix_multiply')
    # print('A: ', A.shape)
    # print('B: ', B.shape)
    # print('block_size: ', block_size)
    # sys.exit(0)
    n = A.shape[0]
    C = torch.zeros((n, n), device=device)
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, n)
                k_end = min(k + block_size, n)
                
                C[i:i_end, j:j_end] += torch.matmul(A[i:i_end, k:k_end], B[k:k_end, j:j_end])
    return C


def matrix_powers_gpu(adj_list, n, block_size, matrix_prefix):
    # 确保节点数 n 可以被块大小 block_size 整除
    assert n % block_size == 0, "n must be divisible by block_size"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 将邻接表转换为邻接矩阵（NumPy 格式）
    adj_matrix_np = create_adjacency_matrix(adj_list, n)
    # 将邻接矩阵转换为 PyTorch 张量，并移动到计算设备上
    adj_matrix = torch.from_numpy(adj_matrix_np).float().to(device)
    file_name = f'{matrix_prefix}1.pkl'
    with open(file_name,'wb') as f:
        pickle.dump(adj_matrix_np,f)
    for k in range(2, 11):
        result_blocks = []

        # 这里的 i 和 j 分别代表行和列的起始索引，这里是将整个图进行了分块处理
        for i in range(0, n, block_size):
            row_blocks = []

            for j in range(0, n, block_size):
                block_shape = (min(i + block_size, n) - i, min(j + block_size, n) - j)
                 # 初始化 A^k 块和 A^(k-1) 块
                A_k_block = torch.eye(*block_shape, device=device)
                A_k_minus_1_block = torch.eye(*block_shape, device=device) if k > 1 else torch.zeros(*block_shape, device=device)
                # 获取当前的邻接矩阵块
                adj_block = adj_matrix[i:i + block_size, j:j + block_size]

                # 计算 A^k 块
                for _ in range(k):
                    A_k_block = block_matrix_multiply(A_k_block, adj_block, block_size, device)
                # 将 A^k 块中的非零元素设置为1
                A_k_block[A_k_block != 0] = 1

                # 如果 k > 1，计算 A^(k-1) 块
                if k > 1:
                    for _ in range(k - 1):
                        A_k_minus_1_block = block_matrix_multiply(A_k_minus_1_block, adj_block, block_size, device)
                    # 将 A^(k-1) 块中的非零元素设置为1
                    A_k_minus_1_block[A_k_minus_1_block != 0] = 1

                # 计算结果块：A^k - A^(k-1)
                result_block = A_k_block - A_k_minus_1_block


                if i == j:
                    result_block += torch.eye(*block_shape, device=device)

                row_blocks.append(result_block.cpu().numpy())


            result_blocks.append(np.concatenate(row_blocks, axis=1))

        full_result = np.concatenate(result_blocks, axis=0)

        # Save the result immediately
        file_name = f'{matrix_prefix}{k}.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(full_result, file)

        # Clear memory
        torch.cuda.empty_cache()

if __name__ == '__main__':
 # generate matrix powers for the adjacency matrix (hogrl)
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "methods/hogrl"))
    from hogrl_utils import filelist, file_matrix_prefix
    
    DATADIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "data/")
    

    for filename, matrix_prefix in zip(filelist.values(), file_matrix_prefix.values()):
        print('generating matrix for: ', filename)
        print('matrix prefix: ', matrix_prefix)
        
        filepath = os.path.join(DATADIR, filename)
        matrix_prefix = os.path.join(DATADIR, matrix_prefix)

        with open(filepath, 'rb') as file:
            relation1 = pickle.load(file)
        
        # print('relation1: ', relation1)
        # 11944 / 8 = 1493 所以对于 amazon 数据集，block_size = 1493 分为 8 块
        # print('len(relation1): ', len(relation1))
        # sys.exit(0)
        
        block_size = 1493 if filename.startswith('amz') else 7659
        matrix_powers_gpu(relation1,len(relation1),block_size,matrix_prefix)