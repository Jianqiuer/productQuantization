from copy import deepcopy
import numpy as np
from heapq import heappush, heappop

def query(queries, codebooks, codes, T):
	P = codebooks.shape[0]
	out = []
	for i in range(len(queries)):
		p_list = np.split(queries[i], P)
		multi_index = np.zeros((P, 256)).astype(np.uint8)
		dis_mtr = np.zeros((P, 256)).astype(np.float32)
		# loop for each product
		for j in range(len(p_list)):
			query_point = p_list[j]
			# calculate L1 distance
			dis_mtr[j] = np.linalg.norm(query_point - codebooks[j], axis=1, ord=1)
			multi_index[j] = (np.argsort(dis_mtr[j]))
		# init search dims for the inverted index
		dims = [0 for _ in range(P)]
		travelled = []
		queue = []
		index_list = [i for i in range(P)]
		# distance to the [0,0] centroid
		dist = sum([dis_mtr[i][multi_index[i][0]] for i in range(len(multi_index))])
		heappush(queue, [dist, dims])
		res = set()
		while len(res) < T:
			dims = heappop(queue)[1]
			travelled.append(dims)
			s = (index_list, dims)
			p_matrix = multi_index[s].astype(np.uint8)
			# find out the code index in the poped centroid
			location = [True for _ in range(len(codes))]
			for l in range(P):
				location &= codes[:, l] == p_matrix[l]

			compare_can = np.where(location)[0].tolist()
			if compare_can:
				for can in compare_can:
					res.add(can)

			if not len(res) < T:
				break

			for e in range(len(dims)):
				if dims[e] < 255:  # K =256
					dims_first = deepcopy(dims)
					dims_first[e] += 1
					add_label = 1
					distance = dis_mtr[e][multi_index[e][dims_first[e]]]
					for i in range(len(dims)):
						if i != e:
							dims_first[i] -= 1
							if dims_first not in travelled and dims_first[i] != -1:
								dims_first[i] += 1
								add_label = 0
								break
							dims_first[i] += 1
							distance += dis_mtr[i][multi_index[i][dims_first[i]]]
					if add_label:
						heappush(queue, [distance, dims_first])
		out.append(res)
	return out


def pq(data, P, init_centroids, max_iter):
	p_list = np.split(data, P, 1)
	codes = np.zeros(shape=(data.shape[0], P))
	for i in range(P):
		init_centroid = init_centroids[i]
		Dataset = p_list[i]  # [N, M/P]

		cluster = np.mat(np.zeros((data.shape[0], 2)))
		for _ in range(max_iter):
			# 1. find the closest centroid
			for j in range(len(Dataset)):
				point = Dataset[j]
				closest_centr = np.argmin(np.abs(point - init_centroid).sum(1))
				cluster[j, :] = closest_centr, 0
			for p in range(256):
				pointsInCluster = Dataset[np.nonzero(cluster[:, 0].A == p)[0]]
				if pointsInCluster.size:
					init_centroid[p, :] = np.median(pointsInCluster, axis=0)
		# update last cluster
		for j in range(len(Dataset)):
			point = Dataset[j]
			closest_centr = np.argmin(np.abs(point - init_centroid).sum(1))
			cluster[j, :] = closest_centr, 0
		codes[:,i] = cluster[:, 0].squeeze()
	return np.float32(init_centroids), np.uint8(codes)



