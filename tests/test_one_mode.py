# import unittest
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def uniformrefine(node, elem, bd_flag=np.array([])):
#     # 3-D case, 三维情况
#     # if np.shape(node)[1] == 3 and np.shape(elem)[1] == 4:  # 3-D
#     #     node, elem, bd_flag, HB = uniformrefine3(node, elem, bd_flag)
#     #     return
#
#     # Construct data structure, 构建数据结构
#     total_edge = np.sort(
#         [elem[:, [1, 2]],
#          elem[:, [2, 0]],
#          elem[:, [0, 1]]],
#         key=lambda _: _[1]  # TODO: 可能有错误
#     ).astype(int)
#
#     edge, temp_var, j = np.unique(total_edge)
#
#     N = np.shape(node)[0]
#     NT = np.shape(elem)[0]
#     NE = np.shape(edge)[0]
#     elem2edge = np.reshape(j, NT, 3).astype(int)
#
#     # Add new nodes: middle points of all edges
#     # 添加新节: 所有边的中点
#     node[N+1:N+NE, :] = (node[edge[:, 0], :] + node[edge[:, 1], :]) / 2
#
#     HB = np.zeros((np.shape(edge)[0], 3))
#     HB[:, [0, 1, 2]] = np.concatenate([
#         np.arange(N+1, N+NE+1).transpose(),
#         edge[:, [0, 1]]
#     ])
#     edge2new_node = np.arange(N+1, N+NE+1).transpose().astype(int)
#
#     # Refine each triangle into four triangles as follows
#     # 将每个三角形细化为四个三角形, 如下所示
#     # 3
#     # | \
#     # 5 - 4
#     # | \ | \
#     # 1 - 6 - 2
#     t = np.arange(NT+1)
#     p = np.zeros((len(t), 6))
#     p[t, [1, 2, 3]] = elem[t, [1, 2, 3]]
#     p[t, [4, 5, 6]] = edge2new_node[elem2edge[t, [1, 2, 3]]]
#     elem[t, :] = np.concatenate([p[t, 0], p[t, 5], p[t, 4]])
#     elem[range(NT+1, 2*NT+1), :] = np.concatenate([p[t, 5], p[t, 1], p[t, 3]])
#     elem[range(2*NT+1, 3*NT+1), :] = np.concatenate([p[t, 4], p[t, 3], p[t, 2]])
#     elem[range(3*NT + 1, 4*NT + 1), :] = np.concatenate([p[t, 3], p[t, 4], p[t, 5]])
#
#     # Update boundary edges
#     # 更新边界节点
#     if len(bd_flag) != 0:
#         bd_flag[range(NT, 2*NT), [0, 2]] = bd_flag[t, [0, 2]]
#         bd_flag[range(2*NT, 3*NT), [0, 1]] = bd_flag[t, [0, 1]]
#         bd_flag[range(3*NT, 4*NT), 0] = 0
#         bd_flag[t, 0] = 0
#     return node, elem, HB, bd_flag
#
# def sortelem(elem, bd_flag=np.array([])):
#     """
#     [elem,bdFlag] = SORTELEM(elem,bdFlag) sorts the elem such that
#     elem(t,1)< elem(t,2)< elem(t,3). A simple sort(elem,2) cannot
#     sort bdFlag.
#     """
#
#     # Step 1: make elem[:, 2] to be the biggest one
#     # 第一步: 将 elem[:, 2] 变成最大的一项
#     temp_var, idx = max(elem, [], 2)  # TODO: 可能有问题
#     # idx = np.array(idx)
#     elem[idx==1, [0, 1, 2]] = elem[idx==1, [1, 2, 0]]
#     elem[idx==2, [0, 1, 2]] = elem[idx == 2, [2, 0, 1]]
#     if len(bd_flag) != 0:
#         bd_flag[idx==1, [0, 1, 2]] = bd_flag[idx==1, [1, 2, 0]]
#         bd_flag[idx == 2, [0, 1, 2]] = bd_flag[idx == 2, [2, 0, 1]]
#
#     # Step 2: sort first two vertices such that elem(:,1)<elem(:2)
#     # 第二步: 排序前两个列向量, 让 elem[:, 0] < elem[:, 1]
#     idx = elem[:, 1] < elem[:, 0]
#     elem[idx, [0, 1]] = elem[idx, [1, 0]]
#     if len(bd_flag) != 0:
#         bd_flag[idx, [0, 1]] = bd_flag[idx, [1, 0]]
#
#     return elem, bd_flag
# def maxwell_stiffness_mass_dirichlet(node,elem,mu_r,epsilon_r,k0):
#     # Sort elem to ascend ordering
#     # 将 elem 升序排列
#     elem, _ = sortelem(elem)  # 为了使 [2 3; 1 3; 1 2] 在elem[t, :]是正向的
#
#     # Construct Data Structure
#     # 构建数据结构
#     # NT: 区域单元的个数
#     # NE: 区域边的个数
#     # NN: 区域单元的节点个数
#     elem2edge, edge, _ = dofedge(elem)
#     NT = np.shape(elem)[0]
#     NE = np.shape(edge)[0]
#     NN = np.shape(node)[0]
#     Dlambda, area, _ = grad_basis(node, elem)
#     curl_phi = np.zeros(NT, 3)
#     curl_phi[:, 0] = 2*(Dlambda[:, 0, 1] * Dlambda[:, 1, 2] - Dlambda[:, 1, 1] * Dlambda[:, 0, 2])
#     curl_phi[:, 1] = 2*(Dlambda[:, 0, 0] * Dlambda[:, 1, 2] - Dlambda[:, 1, 0] * Dlambda[:, 0, 2])
#     curl_phi[:, 2] = 2*(Dlambda[:, 0, 0] * Dlambda[:, 1, 1] - Dlambda[:, 1, 0] * Dlambda[:, 0, 1])
#
#     # Assemble Stiff and Mass matrices of the edge element
#     # 组装边缘单元的刚度矩阵和质量矩阵
#
#
# class MyTestCase(unittest.TestCase):
#     def test_one_mode(self):
#         # 数据设计
#         a = 10e-3
#         b = 4e-3
#         node = np.array([[0, 0],
#                          [a, 0],
#                          [a, b],
#                          [0, b]])
#         elem = np.array([[2, 3, 1],
#                          [4, 1, 3]])
#         num_fine = 2
#         s = 10
#         mu_r = 1
#         epsilon_r = 1
#         f = 100e9
#         c0 = 299_792_458
#         omega = 2 * np.pi * f
#         k0 = omega / c0
#         # theta = np.pi / 6
#         # k0 = 2.0958
#
#         # 网格加密, 粗网格, 细网格
#         node_fine = node
#         elem_fine = num_fine
#         for i in range(num_fine):
#             node_fine, elem_fine, _, _ = uniformrefine(node_fine, elem_fine)
#
#         # PEC
#         A_tt, B_tt, B_tz, B_zt, B_zz, node_inter, edge_inter = maxwell_stiffness_mass_dirichlet(node_Fine,elem_Fine,mu_r,epsilon_r,k0)
#
# if __name__ == '__main__':
#     unittest.main()
