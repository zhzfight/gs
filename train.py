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
    GraphSage,GRUModel
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss, adj_list, split_list, random_walk_with_restart,\
    split_list_by_ratio


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
        train_df['timestamp'] = pd.to_datetime(train_df['local_time']).astype('int64') // 10 ** 9
        val_df['timestamp'] = pd.to_datetime(val_df['local_time']).astype('int64') // 10 ** 9

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
    poi_ids = nodes_df['node_name/poi_id'].tolist()
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))

    # Cat id to index
    cat_ids = list(set(nodes_df[args.feature2].tolist()))
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))

    # Poi idx to cat idx
    poi_idx2cat_idx_dict = {}
    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = \
            cat_id2idx_dict[row[args.feature2]]

    # User id to index
    user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))

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

                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i],ts[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1],ts[i+1]))

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
                    input_seq.append((poi_idxs[i], time_feature[i],ts[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1],ts[i+1]))

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
    seq_model = TransformerModel(num_pois,
                                 num_cats,
                                 args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 dropout=args.transformer_dropout)


    # Define overall loss and optimizer
    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                  list(time_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(embed_fuse_model1.parameters()) +
                                  list(embed_fuse_model2.parameters()) +
                                  list(seq_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_time = maksed_mse_loss

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # %% Tool functions for training
    def input_traj_to_embeddings(sample, poi_embeddings, embedding_index):
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

        poi_idxs=[embedding_index+idx for idx in range(len(input_seq))]
        sample_poi_embedding=poi_embeddings[poi_idxs].squeeze()
        sample_time_embedding=time_embed_model(torch.tensor(input_seq_time,dtype=torch.float).to(device=args.device)).squeeze()
        sample_cat_embedding=cat_embed_model(torch.tensor(input_seq_cat,dtype=torch.long).to(device=args.device)).squeeze()
        sample_fused_embedding1=embed_fuse_model1(user_embedding.expand(len(input_seq),-1),sample_poi_embedding)
        sample_fused_embedding2=embed_fuse_model2(sample_time_embedding,sample_cat_embedding)
        sample_concat_embedding=torch.cat((sample_fused_embedding1,sample_fused_embedding2),dim=-1)

        # POI to embedding and fuse embeddings
        '''
        input_seq_embed = []
        for idx in range(len(input_seq)):
            poi_embedding = poi_embeddings[embedding_index+idx]
            poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)

            # Time to vector
            time_embedding = time_embed_model(
                torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
            time_embedding = torch.squeeze(time_embedding).to(device=args.device)

            # Categroy to embedding
            cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=args.device)
            cat_embedding = cat_embed_model(cat_idx)
            cat_embedding = torch.squeeze(cat_embedding)

            # Fuse user+poi embeds
            fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)
            fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)

            # Concat time, cat after user+poi
            concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)

            # Save final embed
            input_seq_embed.append(concat_embedding)
        '''
        return sample_concat_embedding


    # %% ====================== Train ======================
    poi_embed_model = poi_embed_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    seq_model = seq_model.to(device=args.device)

    # %% Loop epoch
    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_time_loss_list = []
    train_epochs_cat_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_time_loss_list = []
    val_epochs_cat_loss_list = []
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

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_time_loss_list = []
        train_batches_cat_loss_list = []
        #src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            #if len(batch) != args.batch:
                #src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []

            pois = [each[0] for sample in batch for each in sample[1]]
            poi_embeddings = poi_embed_model(torch.tensor(pois).to(args.device))
            # Convert input seq to embeddings

            embedding_index=0
            for sample in batch:
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                input_seq_embed = input_traj_to_embeddings(sample, poi_embeddings,embedding_index)
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
                embedding_index+=len(input_seq)


            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(x)


            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)

            # Final loss
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            train_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

            # Report training progress
            if (b_idx % (40)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                             f'train_batch_loss:{loss.item():.2f}, '
                             f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                             f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                             f'train_move_time_loss:{np.mean(train_batches_time_loss_list):.2f}\n'
                             f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                             f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                             f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                             f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                             f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                             f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq: {batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                             f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)

        # train end --------------------------------------------------------------------------------------------------------
        poi_embed_model.eval()
        user_embed_model.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        seq_model.eval()
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        val_batches_time_loss_list = []
        val_batches_cat_loss_list = []
        #src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        for vb_idx, batch in enumerate(val_loader):
            #if len(batch) != args.batch:
                #src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []

            pois = [each[0] for sample in batch for each in sample[1]]
            poi_embeddings = poi_embed_model(torch.tensor(pois).to(args.device))
            embedding_index=0
            # Convert input seq to embeddings
            for sample in batch:
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                input_seq_time = [each[1] for each in sample[1]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                input_seq_embed = input_traj_to_embeddings(sample, poi_embeddings,embedding_index)
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))
                embedding_index+=len(input_seq)


            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(x)


            # Calculate loss
            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
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
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            val_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            val_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

            # Report validation progress
            if (vb_idx % (20)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                             f'val_batch_loss:{loss.item():.2f}, '
                             f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                             f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                             f'val_move_time_loss:{np.mean(val_batches_time_loss_list):.2f} \n'
                             f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                             f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                             f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                             f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                             f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                             f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq:{batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                             f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                             f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                             '=' * 100)
        # valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_train_time_loss = np.mean(train_batches_time_loss_list)
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        epoch_val_time_loss = np.mean(val_batches_time_loss_list)
        epoch_val_cat_loss = np.mean(val_batches_cat_loss_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_time_loss_list.append(epoch_train_time_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)
        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        val_epochs_time_loss_list.append(epoch_val_time_loss)
        val_epochs_cat_loss_list.append(epoch_val_cat_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)

        # Monitor loss and score
        monitor_loss = epoch_val_loss
        monitor_score = np.mean(epoch_val_top1_acc * 4 + epoch_val_top20_acc)

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
                     f"train_time_loss:{epoch_train_time_loss:.4f}, "
                     f"train_cat_loss:{epoch_train_cat_loss:.4f}, "
                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mAP20:{epoch_train_mAP20:.4f}, "
                     f"train_mrr:{epoch_train_mrr:.4f}\n"
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"val_poi_loss: {epoch_val_poi_loss:.4f}, "
                     f"val_time_loss: {epoch_val_time_loss:.4f}, "
                     f"val_cat_loss: {epoch_val_cat_loss:.4f}, "
                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mAP20:{epoch_val_mAP20:.4f}, "
                     f"val_mrr:{epoch_val_mrr:.4f}")



        # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
            print(f'train_epochs_time_loss_list={[float(f"{each:.4f}") for each in train_epochs_time_loss_list]}',
                  file=f)
            print(f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
            print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                  file=f)
            print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                  file=f)
            print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
            print(f'val_epochs_time_loss_list={[float(f"{each:.4f}") for each in val_epochs_time_loss_list]}', file=f)
            print(f'val_epochs_cat_loss_list={[float(f"{each:.4f}") for each in val_epochs_cat_loss_list]}', file=f)
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
