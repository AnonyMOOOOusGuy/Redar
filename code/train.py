# -*- coding: utf-8 -*-

import os
from utils import print_formatted_current_time
import numpy as np
import torch
from model import Redar
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score
from sklearn.metrics import recall_score
import pandas as pd
from datetime import datetime


def train(args, data_info, model_path):
    train_loader = data_info['train']
    val_loader = data_info['val']
    test_loader = data_info['test']
    feature_num = data_info['feature_num']
    train_num, val_num, test_num = data_info['data_num']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Redar(args, feature_num, device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        weight_decay=args.l2_weight,
        lr=args.lr
    )
    crit = torch.nn.BCELoss()

    print_formatted_current_time('Model training begins...')


    for step in range(args.n_epoch):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        if step % 5 == 0:
            print(f"[{formatted_time}] epoch[{step + 1}/{args.n_epoch}]", end=" ")
        loss_all = 0
        model.train()
        for data in train_loader:
            data = data.to(device)
            output = model(data)
            label = data.y
            label = label.to(device)
            baseloss = crit(torch.squeeze(output), label)
            loss = baseloss
            loss_all += data.num_graphs * loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cur_loss = loss_all / train_num
        val_auc, val_logloss, val_ndcg5, val_ndcg10, val_recall = evaluate(model, val_loader, device)
        test_auc, test_logloss, test_ndcg5, test_ndcg10, test_recall = evaluate(model, test_loader, device)

        model.cur_loss_list.append(cur_loss)
        model.val_auc_list.append(val_auc)
        model.test_auc_list.append(test_auc)
        model.val_ndcg10_list.append(val_ndcg10)
        model.test_ndcg10_list.append(test_ndcg10)
        model.val_recall_list.append(val_recall)
        model.test_recall_list.append(test_recall)

        if step % 5 == 0:
            print(f"Loss: {cur_loss:.6f};"
                  f" val_AUC: {val_auc:.6f} test_AUC: {test_auc:.6f};"
                  f" val_NDGC@5: {val_ndcg5:.6f} test_NDGC@5: {test_ndcg5:.6f}; "
                  f" val_recall: {val_recall:.6f} test_recall: {test_recall:.6f}")

    # save the model
    if not os.path.exists(model_path):
        os.makedirs('..\\checkpoint\\')
    torch.save(model, model_path)

    print(f"The model training has been completed, and the training results have been saved to {model_path}")






def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    labels = []
    user_ids = []
    with torch.no_grad():
        for data in data_loader:
            _, user_id_index = np.unique(data.batch.detach().cpu().numpy(), return_index=True)
            user_id = data.x.detach().cpu().numpy()[user_id_index]
            user_ids.append(user_id)

            data = data.to(device)
            pred = model(data)
            pred = pred.squeeze().detach().cpu().numpy().astype('float64')
            if pred.size == 1:
                pred = np.expand_dims(pred, axis=0)
            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.concatenate(predictions, 0)
    labels = np.concatenate(labels, 0)
    user_ids = np.concatenate(user_ids, 0)

    ndcg5 = cal_ndcg(predictions, labels, user_ids, 5)
    ndcg10 = cal_ndcg(predictions, labels, user_ids, 10)
    auc = roc_auc_score(labels, predictions)
    logloss = log_loss(labels, predictions)

    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)
    recall = recall_score(labels, binary_predictions, average='micro')

    return auc, logloss, ndcg5, ndcg10, recall


def cal_ndcg(predicts, labels, user_ids, k):
    df = pd.DataFrame(
        {'user': np.squeeze(user_ids),
         'predict': np.squeeze(predicts),
         'label': np.squeeze(labels)
         })

    ndcg = []
    for user_id, user_srow in df.groupby('user'):
        upred = user_srow['predict'].tolist()
        if len(upred) < 2:
            continue
        ulabel = user_srow['label'].tolist()
        ndcg.append(ndcg_score([ulabel], [upred], k=k))

    return np.mean(np.array(ndcg))
