""" Build the user-agnostic global trajectory flow map from the sequence data """
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans

def build_global_POI_checkin_graph(df, exclude_user=None):
    G = nx.DiGraph()
    users = list(set(df['user_id'].to_list()))
    if exclude_user in users: users.remove(exclude_user)
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user_id'] == user_id]

        # Add node (POI)
        for i, row in user_df.iterrows():
            node = row['POI_id']
            if node not in G.nodes():
                G.add_node(row['POI_id'],
                           checkin_cnt=1,
                           poi_catid=row['POI_catid'],
                           poi_catid_code=row['POI_catid_code'],
                           poi_catname=row['POI_catname'],
                           latitude=row['latitude'],
                           longitude=row['longitude'])
            else:
                G.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in seq)
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['POI_id']
            traj_id = row['trajectory_id']
            # No edge for the begin of the seq or different traj
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue

            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_traj_id = traj_id
            previous_poi_id = poi_id

    return G

# 定义一个函数来计算两个点之间的欧几里得距离
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)




def save_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.

    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    # np.save(os.path.join(dst_dir, 'adj_mtx.npy'), A.todense())
    np.savetxt(os.path.join(dst_dir, 'graph_A.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, 'graph_X.csv'), 'w') as f:
        print('node_name/poi_id,checkin_cnt,poi_catid,poi_catid_code,poi_catname,latitude,longitude', file=f)
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            poi_catid = each[1]['poi_catid']
            poi_catid_code = each[1]['poi_catid_code']
            poi_catname = each[1]['poi_catname']
            latitude = each[1]['latitude']
            longitude = each[1]['longitude']
            print(f'{node_name},{checkin_cnt},'
                  f'{poi_catid},{poi_catid_code},{poi_catname},'
                  f'{latitude},{longitude}', file=f)
    # 读取数据
    df = pd.read_csv(os.path.join(dst_dir, 'graph_X.csv'),sep=',')
    # 提取经纬度特征
    X = df[["longitude", "latitude"]]
    # 创建K-MEANS模型，假设聚类数为4
    model = KMeans(n_clusters=100, random_state=0)
    # 拟合模型
    model.fit(X)
    # 预测类别
    labels = model.predict(X)
    # 在pandas后面添加一列表示类别
    df["cluster"] = labels
    # 使用pandas.apply方法来对数据框中的每一行应用这个函数，并得到一个新的列
    # 定义一个函数来计算一个POI与其他POI之间的距离，并返回最近的4个cluster
    # 使用pandas.groupby方法来对数据框按照cluster进行分组，并计算每个分组的坐标的平均值，这样就得到了每个cluster的质心
    centroids = df.groupby('cluster').agg({'longitude': 'mean', 'latitude': 'mean'}).reset_index()

    # 定义一个函数来计算一个POI与其他cluster的质心之间的距离，并返回最近的4个cluster
    def nearest_clusters(POI_ID, centroids, k=4):
        # 获取给定POI的坐标和cluster
        x = df[df['node_name/poi_id'] == POI_ID]['longitude'].values[0]
        y = df[df['node_name/poi_id'] == POI_ID]['latitude'].values[0]
        cluster = df[df['node_name/poi_id'] == POI_ID]['cluster'].values[0]
        # 创建一个空列表来存储最近cluster和距离
        nearest = []
        # 遍历其他cluster
        for c in centroids['cluster'].unique():
            # 跳过给定POI所属的cluster
            if c == cluster:
                continue
            # 获取其他cluster的质心坐标
            x_other = centroids[centroids['cluster'] == c]['longitude'].values[0]
            y_other = centroids[centroids['cluster'] == c]['latitude'].values[0]
            # 计算两个点之间的距离
            dist = euclidean_distance(x, y, x_other, y_other)
            # 将cluster和距离添加到列表中
            nearest.append((c, dist))
        # 对列表按照距离进行排序
        nearest.sort(key=lambda x: x[1])
        # 返回最近的k个cluster，只保留cluster编号，不保留距离
        return [x[0] for x in nearest[:k]]

    # 使用pandas.apply方法来对数据框中的每一行应用这个函数，并得到一个新的列
    df['nearest_clusters'] = df['node_name/poi_id'].apply(nearest_clusters, args=(centroids,))
    df.to_csv(os.path.join(dst_dir, 'graph_X.csv'),index=False)


def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, 'graph.pkl'), 'wb'))


def save_graph_edgelist(G, dst_dir):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join(dst_dir, 'graph_node_id2idx.txt'), 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node}, {i}', file=f)

    with open(os.path.join(dst_dir, 'graph_edge.edgelist'), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[src_node]} {node_id2idx[dst_node]} {weight}', file=f)


def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude', feature4='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X


def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    # Node degrees (mean and percentiles)
    node_degrees = [each[1] for each in G.degree]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    # Edge weights (mean and percentiles)
    edge_weights = []
    for n, nbrs in G.adj.items():
        for nbr, attr in nbrs.items():
            weight = attr['weight']
            edge_weights.append(weight)
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")




if __name__ == '__main__':
    dst_dir = r'dataset/NYC'

    # Build POI checkin trajectory graph
    train_df = pd.read_csv(os.path.join(dst_dir, 'NYC_train.csv'))
    print('Build global POI checkin graph -----------------------------------')
    G = build_global_POI_checkin_graph(train_df)

    # Save graph to disk
    save_graph_to_pickle(G, dst_dir=dst_dir)
    save_graph_to_csv(G, dst_dir=dst_dir)
    save_graph_edgelist(G, dst_dir=dst_dir)
