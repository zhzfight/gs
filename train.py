import collections
import logging
import logging
import multiprocessing
import os
import pathlib
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import load_graph_adj_mtx, load_graph_node_features
from model import GCN, NodeAttnMap, UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings, TransformerModel, \
    GraphSage,TimeAwareTransformer,Projection
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss, adj_list, split_list, random_walk_with_restart, \
    split_list_by_ratio, neg_sample


def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)



    if args.data_train!='dataset/dataset_tsmc2014/NYC_train.csv':
        train_df['timestamp'] = pd.to_datetime(train_df['local_time'])
        train_df['hour'] = train_df['timestamp'].dt.hour
        train_df['timestamp']=train_df['timestamp'].astype('int64') // 10 ** 9
        val_df['timestamp'] = pd.to_datetime(val_df['local_time'])
        val_df['hour'] = val_df['timestamp'].dt.hour
        val_df['timestamp'] = val_df['timestamp'].astype('int64') // 10 ** 9
    # Build POI graph (built from train_df)
    print('Loading POI graph...')
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    raw_X = load_graph_node_features(args.data_node_feats,
                                     args.feature1,
                                     args.feature2,
                                     args.feature3,
                                     args.feature4)
    logging.info(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}.")
    logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")
    num_pois = raw_X.shape[0]

    # One-hot encoding poi categories
    logging.info('One-hot encoding poi categories id')
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")
    logging.info(f'POI categories: {list(one_hot_encoder.categories_[0])}')
    # Save ont-hot encoder
    with open(os.path.join(args.save_dir, 'one-hot-encoder.pkl'), "wb") as f:
        pickle.dump(one_hot_encoder, f)

    # Normalization
    print('Laplician matrix...')
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')

    # POI id to index
    nodes_df = pd.read_csv(args.data_node_feats)
    nodes_df['nearest_clusters'] = nodes_df['nearest_clusters'].apply(eval)
    poi_ids = nodes_df['node_name/poi_id'].tolist()
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))

    # Cat id to index
    cat_ids = list(set(nodes_df[args.feature2].tolist()))
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))

    # Poi idx to cat idx
    poi_idx2cat_idx_dict = {}
    poi_idx2kmeans={}
    poi_idx2nearest_cluster={}
    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = \
            cat_id2idx_dict[row[args.feature2]]
        poi_idx2kmeans[poi_id2idx_dict[row['node_name/poi_id']]]=row['cluster']
        poi_idx2nearest_cluster[poi_id2idx_dict[row['node_name/poi_id']]]=row['nearest_clusters']

    cluster= collections.defaultdict(list)
    for i,row in nodes_df.iterrows():
        cluster[row['cluster']].append(poi_id2idx_dict[row['node_name/poi_id']])
    user_poi_list=train_df.groupby("user_id")["POI_id"].unique().apply(list)
    user_poi_dict = user_poi_list.to_dict()

    # User id to index
    user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))
    u_checkIn_poi={}
    for k,v in user_poi_dict.items():
        v=[poi_id2idx_dict[each] for each in v]
        u_checkIn_poi[user_id2idx_dict[str(k)]]=v
    df = pd.read_csv('dataset/NYC/NYC_train.csv')
    # 把timestamp列转换成pandas的datetime类型
    df['timestamp'] = pd.to_datetime(df['local_time'])

    # 创建一个新的列，叫做hour，用来存储timestamp中的小时部分
    df['hour'] = df['timestamp'].dt.hour
    # 把hour列转换成一个整数除以4的商
    df['hour'] = df['hour'] // 2
    # 使用pandas.groupby方法来对数据框按照poi_id和hour进行分组，并计算每个分组的行数
    df = df.groupby(['POI_id', 'hour']).size().reset_index(name='count')

    # 使用pandas.pivot_table方法来把数据框转换成一个透视表
    df = df.pivot_table(index='POI_id', columns='hour', values='count', fill_value=0)
    # 把df的行索引（poi_id）转换成一列
    df = df.reset_index()

    # 使用pandas.Series.map方法来对poi_id列应用字典，得到一个新的列，叫做new_value
    df['new_value'] = df['POI_id'].map(poi_id2idx_dict)

    # 使用pandas.set_index方法来把new_value列设为新的行索引，并删除poi_id列
    df = df.set_index('new_value').drop('POI_id', axis=1)

    # 使用pandas.DataFrame.to_dict方法来把df转化成字典
    timestatic = df.to_dict(orient='index')


    # Print user-trajectories count
    traj_list = list(set(train_df['trajectory_id'].tolist()))
    print('load adj_list')
    if os.path.exists(os.path.join(args.adj_path, 'adj.pkl')):
        with open(os.path.join(args.adj_path, 'adj.pkl'), 'rb') as f:  # 打开pickle文件
            adj = pickle.load(f)  # 读取字典
        with open(os.path.join(args.adj_path, 'dis.pkl'), 'rb') as f:  # 打开pickle文件
            dis = pickle.load(f)  # 读取字典
    else:
        adj, dis = adj_list(raw_A, raw_X, args.geo_dis)
        with open(os.path.join(args.adj_path, 'adj.pkl'), 'wb') as f:
            pickle.dump(adj, f)  # 把字典写入pickle文件
        with open(os.path.join(args.adj_path, 'dis.pkl'), 'wb') as f:
            pickle.dump(dis, f)  # 把字典写入pickle文件

    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []  # traj id: user id + traj no.
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(train_df['trajectory_id'].tolist())):
                traj_df = train_df[train_df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
                time_feature = traj_df[args.time_feature].to_list()
                ts = traj_df['timestamp'].to_list()
                hour=traj_df['hour'].to_list()
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i],ts[i],hour[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1],ts[i+1],hour[i+1]))

                if len(input_seq) < args.short_traj_thres:
                    continue

                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(df['trajectory_id'].tolist())):
                user_id = traj_id.split('_')[0]

                # Ignore user if not in training set
                if user_id not in user_id2idx_dict.keys():
                    continue

                # Ger POIs idx in this trajectory
                traj_df = df[df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = []
                time_feature = traj_df[args.time_feature].to_list()
                ts = traj_df['timestamp'].to_list()
                hour=traj_df['hour'].to_list()

                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        # Ignore poi if not in training set
                        continue

                # Construct input seq and label seq
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i],ts[i],hour[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1],ts[i+1],hour[i+1]))

                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)

    class produceSampleProcess(multiprocessing.Process):
        def __init__(self, tasks, queues, adj_list, restart_prob, num_walks, threshold, adjOrdis, stop_event, id):
            super().__init__()
            self.tasks = tasks
            self.queues = queues
            self.threshold = threshold
            self.adjOrdis = adjOrdis
            self.stop_event = stop_event
            self.id = id
            self.adj_list = adj_list
            self.restart_prob = restart_prob
            self.num_walks = num_walks
            self.count_dict={key:threshold for key in tasks}
            self.missing_dict={key:0 for key in tasks}

        def run(self):
            while True:
                for node in self.tasks:
                    q = self.queues[node]
                    if q.qsize() < self.threshold/2:
                        for _ in range(self.count_dict[node]-q.qsize()):
                            random_walk = random_walk_with_restart(self.adj_list, node, self.restart_prob, self.num_walks,
                                                               self.adjOrdis)
                            q.put(random_walk)
                        self.missing_dict[node]+=1
                        if self.missing_dict[node]>2:
                            self.missing_dict[node]=0
                            self.count_dict[node]+=self.threshold
                if self.stop_event.is_set():
                    break
            print(self.adjOrdis, self.id, 'quit')

    threshold = 10  # 队列大小阈值
    adj_queues = {node: multiprocessing.Queue() for node in range(num_pois)}  # 创建多个队列
    dis_queues = {node: multiprocessing.Queue() for node in range(num_pois)}  # 创建多个队列
    tasks = split_list([i for i in range(num_pois)],int(args.cpus/2))
    stop_event = multiprocessing.Event()

    for idx, task in enumerate(tasks):
        ap = produceSampleProcess(tasks=task, queues=adj_queues, adj_list=adj, restart_prob=args.restart_prob,
                                  num_walks=args.num_walks,
                                  threshold=threshold, adjOrdis='adj', stop_event=stop_event, id=idx)
        ap.start()
        dp = produceSampleProcess(tasks=task, queues=dis_queues, adj_list=dis, restart_prob=args.restart_prob,
                                  num_walks=args.num_walks,
                                  threshold=threshold, adjOrdis='dis', stop_event=stop_event, id=idx)
        dp.start()

    # %% ====================== Build Models ======================
    # Model1: POI embedding model
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)

    poi_embed_model = GraphSage(X=X,  embed_dim=args.poi_embed_dim, adj=adj, dis=dis,
                                device=args.device, restart_prob=args.restart_prob, num_walks=args.num_walks,
                                dropout=args.dropout, adj_queues=adj_queues, dis_queues=dis_queues)


    # %% Model2: User embedding model, nn.embedding
    num_users = len(user_id2idx_dict)
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)

    # %% Model3: Time Model
    time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)

    # %% Model4: Category embedding model
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)

    # %% Model5: Embedding fusion models
    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
    embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)

    # %% Model6: Sequence model
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
    proj=Projection(args.poi_embed_dim+ args.cat_embed_dim)
    seq_model=TimeAwareTransformer(num_poi=num_pois,
                         num_cat=num_cats,
                         nhid=args.seq_input_embed,
                        output_dim=args.poi_embed_dim+args.cat_embed_dim,
                         batch_size=args.batch,
                         device=args.device,
                         dropout=args.transformer_dropout)


    # Define overall loss and optimizer
    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                  list(time_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(embed_fuse_model1.parameters()) +
                                  list(embed_fuse_model2.parameters()) +
                                  list(seq_model.parameters())+
                                    list(proj.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)




    # %% Tool functions for training
    def input_traj_to_embeddings(sample, poi_embeddings, embedding_index=None):
        # Parse sample
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]

        # User to embedding
        user_id = traj_id.split('_')[0]
        user_idx = user_id2idx_dict[user_id]
        input = torch.LongTensor([user_idx]).to(device=args.device)
        user_embedding = user_embed_model(input)
        user_embedding = torch.squeeze(user_embedding)
        if embedding_index==None:
            poi_idxs=input_seq
        else:
            poi_idxs=[embedding_index+idx for idx in range(len(input_seq))]
        sample_poi_embedding=poi_embeddings[poi_idxs].squeeze()
        sample_time_embedding=time_embed_model(torch.tensor(input_seq_time,dtype=torch.float).to(device=args.device)).squeeze()
        sample_cat_embedding=cat_embed_model(torch.tensor(input_seq_cat,dtype=torch.long).to(device=args.device)).squeeze()
        sample_fused_embedding1=embed_fuse_model1(user_embedding.expand(len(input_seq),-1),sample_poi_embedding)
        sample_fused_embedding2=embed_fuse_model2(sample_time_embedding,sample_cat_embedding)
        sample_concat_embedding=torch.cat((sample_fused_embedding1,sample_fused_embedding2),dim=-1)
        return sample_concat_embedding


    # %% ====================== Train ======================
    poi_embed_model = poi_embed_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    seq_model = seq_model.to(device=args.device)
    proj=proj.to(device=args.device)

    # %% Loop epoch
    # For plotting
    train_epochs_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    # For saving ckpt
    max_val_score = -np.inf

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_embed_model.train()
        user_embed_model.train()
        time_embed_model.train()
        cat_embed_model.train()
        embed_fuse_model1.train()
        embed_fuse_model2.train()
        seq_model.train()
        proj.train()

        train_batches_loss_list = []
        #src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            #if len(batch) != args.batch:
                #src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_input_seqs_ts = []
            batch_label_seqs_ts = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_cat=[]
            batch_seq_n_poi=[]
            batch_seq_n_cat=[]
            k=8

            for sample in batch:
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
                traj_id = sample[0]
                user_id = traj_id.split('_')[0]
                input_seq = [each[0] for each in sample[1]]
                input_seq_ts = [each[2] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                label_hour=[each[3] for each in sample[2]]
                label_cat=[poi_idx2cat_idx_dict[poi] for poi in label_seq]
                n_seq=neg_sample(k,u_checkIn_poi,label_seq,label_hour,timestatic,cluster,poi_idx2nearest_cluster,poi_idx2kmeans)

                n_cat=[poi_idx2cat_idx_dict[poi] for poi in n_seq]
                label_seq_ts = [each[2] for each in sample[2]]
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_input_seqs_ts.append(input_seq_ts)
                batch_label_seqs_ts.append(label_seq_ts)
                batch_seq_labels_poi.append(label_seq)
                batch_seq_labels_cat.append(label_cat)
                batch_seq_n_poi.append(n_seq)
                batch_seq_n_cat.append(n_cat)
            pois=[each[0] for sample in batch for each in sample[1]]
            all_seq_len=len(pois)
            pois.extend([each for sample in batch_seq_labels_poi for each in sample])
            pois.extend([n_poi for seq in batch_seq_n_poi for n_poi in seq ])
            poi_embeddings = poi_embed_model(torch.tensor(pois).to(args.device))
            input_embeddings=poi_embeddings[:all_seq_len]
            label_embeddings=poi_embeddings[all_seq_len:2*all_seq_len]
            ne_embeddings=poi_embeddings[2*all_seq_len:]
            ne_embeddings=ne_embeddings.view(k,all_seq_len,args.poi_embed_dim)
            label_cats=[cat for sample in batch_seq_labels_cat for cat in sample]
            n_cats=[cat for sample in batch_seq_n_cat for cat in sample]
            label_cat_embeddings=cat_embed_model(torch.LongTensor(label_cats).to(args.device))
            n_cat_embeddings=cat_embed_model(torch.LongTensor(n_cats).to(args.device))
            n_cat_embeddings=n_cat_embeddings.view(k,all_seq_len,args.cat_embed_dim)
            label_embeddings=torch.cat((label_embeddings,label_cat_embeddings),dim=1)
            ne_embeddings=torch.cat((ne_embeddings,n_cat_embeddings),dim=2)

            embedding_index = 0
            for idx,sample in enumerate(batch):
                input_seq_embed=input_traj_to_embeddings(sample,input_embeddings,embedding_index)
                batch_seq_embeds.append(input_seq_embed)
                embedding_index+=batch_seq_lens[idx]



            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)


            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            preference = seq_model(x,batch_seq_lens,batch_input_seqs_ts,batch_label_seqs_ts)
            flatten_preference=torch.zeros(all_seq_len,preference.shape[2]).to(args.device)
            start_point=0
            for idx,seq_len in enumerate(batch_seq_lens):
                flatten_preference[start_point:start_point+seq_len,:]=preference[idx,:seq_len,:]
                start_point+=seq_len
            label_embeddings=proj(label_embeddings)
            ne_embeddings=proj(ne_embeddings)
            dot_pos=torch.einsum('ij,ij->i', flatten_preference, label_embeddings)
            dot_ne=torch.einsum('ij,kij->ki', flatten_preference, ne_embeddings)
            sub=torch.sub(dot_pos,dot_ne)
            nll=torch.mean(-torch.log(torch.sigmoid(sub)))
            #reg=args.lambda_theta * (flatten_preference.norm() + label_embeddings.norm() + ne_embeddings.norm())


            # Final loss
            loss = nll
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # Report training progress
            if (b_idx % (100)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                             f'train_batch_loss:{loss.item():.2f}, '+
                             '=' * 100)
        # train end --------------------------------------------------------------------------------------------------------
        poi_embed_model.eval()
        user_embed_model.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        seq_model.eval()
        proj.eval()
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        pois = [n for n in range(num_pois)]
        cats = [poi_idx2cat_idx_dict[n] for n in pois]
        poi_embeddings = poi_embed_model(torch.tensor(pois).to(args.device))
        cat_embeddings = cat_embed_model(torch.tensor(cats).to(args.device))
        l_embeddings = torch.cat((poi_embeddings, cat_embeddings), dim=1)
        l_embeddings=proj(l_embeddings)


        #src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        for vb_idx, batch in enumerate(val_loader):
            #if len(batch) != args.batch:
                #src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_input_seqs_ts = []
            batch_label_seqs_ts = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []

            # Convert input seq to embeddings
            for sample in batch:
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                input_seq_ts = [each[2] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                label_seq_ts = [each[2] for each in sample[2]]

                input_seq_embed = input_traj_to_embeddings(sample, poi_embeddings)
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_input_seqs_ts.append(input_seq_ts)
                batch_label_seqs_ts.append(label_seq_ts)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))


            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            preference= seq_model(x,batch_seq_lens,batch_input_seqs_ts,batch_label_seqs_ts)
            y_pred_poi=torch.matmul(preference,l_embeddings.transpose(1,0))


            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))

            # Report validation progress
            if (vb_idx % (200)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                             f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                             f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                             f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                             f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                             f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                             f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq:{batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)
        # valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)


        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)


        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)


        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mAP20:{epoch_val_mAP20:.4f}, "
                     f"val_mrr:{epoch_val_mrr:.4f}")



        # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
            print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)


if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    train(args)
