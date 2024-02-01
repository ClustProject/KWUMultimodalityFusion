import numpy as np
from .MachineLearning import kneighbors
import time

def ssm_fusion(ssm_1, ssm_2, nssm_1, nssm_2, k, t):
    #print("\n********** Local SSM fusion start ***********")
    length = ssm_1.shape[0]

    skm_1 = np.zeros((length, length), dtype='float64')  # km means kernel matrix
    skm_2 = np.zeros((length, length), dtype='float64')

    f1_neighbors = kneighbors(ssm_1, length, k)
    f2_neighbors = kneighbors(ssm_2, length, k)

    #print("sparse kernel matrix construction start...")
    # 1st feature based sparse kernel matrix construction
    for i in range(length):
        f1_ith_neighs = f1_neighbors[i]
        skm_1[i][f1_ith_neighs] = ssm_1[i][f1_ith_neighs] / np.sum(ssm_1[i][f1_ith_neighs])

        f2_ith_neighs = f2_neighbors[i]
        skm_2[i][f2_ith_neighs] = ssm_2[i][f2_ith_neighs] / np.sum(ssm_2[i][f2_ith_neighs])
    #print("1st feature based skm has been completed")
    #print("2nd feature based skm has been completed\n")

    #print("fused ssm construction start...")
    # make normalized weight matrices by iterating t times

    st = time.time()

    for _t in range(t):
        #print("time step : ", _t)
        temp = nssm_1.copy()
        nssm_1 = np.matmul(np.matmul(skm_1, nssm_2.copy()), skm_1.T)
        nssm_2 = np.matmul(np.matmul(skm_2, temp), skm_2.T)

    fused_ssm = (nssm_1 + nssm_2) / 2

    #print(f"{time.time() - st:.4f} sec")  # 종료와 함께 수행시간 출력
    #print("Done")
    #print("**********************************************")
    return fused_ssm


def global_ssm_fusion(fsm1, fsm2, fsm3, k, t):
    #print("\n********** Global SSM fusion start ***********")
    length = fsm1.shape[0]

    skm_1 = np.zeros((length, length), dtype='float64')  # km means kernel matrix
    skm_2 = np.zeros((length, length), dtype='float64')
    skm_3 = np.zeros((length, length), dtype='float64')

    f1_neighbors = kneighbors(fsm1, length, k)
    f2_neighbors = kneighbors(fsm2, length, k)
    f3_neighbors = kneighbors(fsm3, length, k)

    #print("sparse kernel matrix construction start...")
    # 1st feature based sparse kernel matrix construction
    for i in range(length):
        f1_ith_neighs = f1_neighbors[i]
        skm_1[i][f1_ith_neighs] = fsm1[i][f1_ith_neighs] / np.sum(fsm1[i][f1_ith_neighs])

        f2_ith_neighs = f2_neighbors[i]
        skm_2[i][f2_ith_neighs] = fsm2[i][f2_ith_neighs] / np.sum(fsm2[i][f2_ith_neighs])

        f3_ith_neighs = f3_neighbors[i]
        skm_3[i][f3_ith_neighs] = fsm3[i][f3_ith_neighs] / np.sum(fsm3[i][f3_ith_neighs])

    #print("Three skms has been completed")

    #print("fused ssm construction start...")

    # st = time.time()

    for _t in range(t):
        #print("time step : ", _t)
        fsm1_copy = fsm1.copy()
        fsm2_copy = fsm2.copy()
        fsm3_copy = fsm3.copy()

        fsm1 = np.matmul(np.matmul(skm_1, (fsm2_copy + fsm3_copy) / 2), skm_1.T)
        fsm2 = np.matmul(np.matmul(skm_2, (fsm1_copy + fsm3_copy) / 2), skm_2.T)
        fsm3 = np.matmul(np.matmul(skm_3, (fsm1_copy + fsm2_copy) / 2), skm_3.T)

    fused_ssm = (fsm1 + fsm2 + fsm3) / 3

    #print(f"{time.time() - st:.4f} sec")  # 종료와 함께 수행시간 출력
    #print("Done")
    #print("**********************************************")
    return fused_ssm

def concatenate_fusion(*args):
    fused_feature = args[0]
    for arg in args[1:]:
        fused_feature = np.concatenate((fused_feature, arg), axis = 1 )
    return fused_feature