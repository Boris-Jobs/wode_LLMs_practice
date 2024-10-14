# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt


def init_network(model, method="xavier", exclude="embedding", seed=42):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if "weight" in name:
                if method == "xavier":
                    nn.init.xavier_normal_(w)
                elif method == "kaiming":
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif "bias" in name:
                nn.init.constant_(w, 0)
            else:
                pass


train_losses = []
train_topic_accs = []
train_emo_accs = []
test_losses = []
test_topic_accs = []
test_emo_accs = []


def train(config, model, train_iter, dev_iter, test_iter):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)


    for epoch in range(config.num_epochs):
        total_correct_topic = 0
        total_correct_emo = 0
        total_samples = 0
        total_batch = 0
        all_preds_topic = []
        all_labels_topic = []
        all_preds_emo = []
        all_labels_emo = []
        train_topic_acc = 0
        train_emo_acc = 0

        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        for (trains, labels, emotion_scores) in train_iter:
            loss, correct_topic, correct_emo, pred_topic, pred_emo = loss_correct(trains, labels, emotion_scores, model)
            model.zero_grad()
            total_correct_topic += correct_topic
            total_correct_emo += correct_emo
            total_samples += labels.size(0)
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            # 收集预测和真实标签
            all_preds_topic.extend(pred_topic.cpu().numpy())
            all_labels_topic.extend(labels.cpu().numpy())
            all_preds_emo.extend(pred_emo.cpu().numpy())
            all_labels_emo.extend(emotion_scores.cpu().numpy())
            if total_batch == 75:
                # 计算训练准确率
                print("Now, the epoch is: ", epoch, "and the total batch is: ", total_batch)
                train_topic_acc = total_correct_topic / total_samples
                train_emo_acc = total_correct_emo / total_samples

                # 计算F1、精确率、召回率
                train_f1_topic, train_precision_topic, train_recall_topic, _ = calculate_metrics(all_labels_topic, all_preds_topic, average='macro')
                train_f1_emo, train_precision_emo, train_recall_emo, _ = calculate_metrics(all_labels_emo, all_preds_emo, average='macro')

                dev_topic_acc, dev_emo_acc, dev_loss = evaluate(config, model, dev_iter)

                msg = ("Iter: {0:>6},  Train Loss: {1:>5.2},  Train Topic Acc: {2:>6.2%},  Train Emo Acc: {3:>6.2%}, "
                       "Train F1 Topic: {4:>6.2%}, Train Precision Topic: {5:>6.2%}, Train Recall Topic: {6:>6.2%}, "
                       "Train F1 Emo: {7:>6.2%}, Train Precision Emo: {8:>6.2%}, Train Recall Emo: {9:>6.2%}, "
                       "Val Loss: {10:>5.2},  Val Topic Acc: {11:>6.2%},  Val Emo Acc: {12:>6.2%}")
                print(msg.format(
                        total_batch,
                        loss.item(),
                        train_topic_acc,
                        train_emo_acc,
                        train_f1_topic,
                        train_precision_topic,
                        train_recall_topic,
                        train_f1_emo,
                        train_precision_emo,
                        train_recall_emo,
                        dev_loss,
                        dev_topic_acc,
                        dev_emo_acc))
            total_batch += 1
        scheduler.step()
        torch.save(model.state_dict(), config.save_path)
        test(config, model, test_iter)
        train_losses.append(loss.item())
        train_topic_accs.append(train_topic_acc)
        train_emo_accs.append(train_emo_acc)
    # 在训练完成后调用 plot_metrics 函数
    plot_metrics(train_losses, train_topic_accs, train_emo_accs, test_losses, test_topic_accs, test_emo_accs, save_path='trainint_metrics.png')



def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    all_preds_topic = []
    all_labels_topic = []
    all_preds_emo = []
    all_labels_emo = []
    total_loss = 0
    with torch.no_grad():
        for (x, labels, emotion_scores) in test_iter:
            loss, _, _, pred_topic, pred_emo = loss_correct(x, labels, emotion_scores, model)
            all_preds_topic.extend(pred_topic.cpu().numpy())
            all_labels_topic.extend(labels.cpu().numpy())
            all_preds_emo.extend(pred_emo.cpu().numpy())
            all_labels_emo.extend(emotion_scores.cpu().numpy())
            total_loss += loss.item()
    test_f1_topic, test_precision_topic, test_recall_topic, test_topic_acc = calculate_metrics(all_labels_topic, all_preds_topic, average='macro')
    test_f1_emo, test_precision_emo, test_recall_emo, test_emo_acc = calculate_metrics(all_labels_emo, all_preds_emo, average='macro')

    msg = ("Test F1 Topic: {0:>6.2%}, Test Precision Topic: {1:>6.2%}, Test Recall Topic: {2:>6.2%}, Test Topic Acc: {3:>6.2%}, "
           "Test F1 Emo: {4:>6.2%}, Test Precision Emo: {5:>6.2%}, Test Recall Emo: {6:>6.2%}, Test Emo Acc: {7:>6.2%}")
    print(msg.format(
            test_f1_topic,
            test_precision_topic,
            test_recall_topic,
            test_topic_acc,
            test_f1_emo,
            test_precision_emo,
            test_recall_emo,
            test_emo_acc))
    test_losses.append(total_loss)
    test_topic_accs.append(test_topic_acc)
    test_emo_accs.append(test_emo_acc)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    total_correct_topic = 0
    total_correct_emo = 0
    total_samples = 0
    with torch.no_grad():
        for (x, labels, emotion_scores) in data_iter:
            loss, correct_topic, correct_emo, _, _ = loss_correct(x, labels, emotion_scores, model)
            loss_total += loss.item()

            total_correct_topic += correct_topic
            total_correct_emo += correct_emo
            batch_size = labels.size(0)
            total_samples += batch_size

    topic_acc = total_correct_topic / total_samples
    emo_acc = total_correct_emo / total_samples
    return topic_acc, emo_acc, loss_total


def loss_correct(x, labels, emotion_scores, model):
    out_topic, out_emo = model(x)  # 获取情感和主题的输出

    # 计算损失
    true_topic = labels.float()  # 保持在 GPU 上
    pred_prob_topic = torch.sigmoid(out_topic)  # 将输出转换为概率
    pred_topic = (pred_prob_topic > 0.5).float()

    true_emo = emotion_scores.float()  # 保持在 GPU 上
    pred_prob_emo = torch.sigmoid(out_emo)  # 将输出转换为概率
    pred_emo = torch.zeros_like(pred_prob_emo)
    max_indices = torch.argmax(pred_prob_emo, dim=1)
    pred_emo.scatter_(1, max_indices.unsqueeze(1), 1)

    loss_topic = F.binary_cross_entropy_with_logits(out_topic, true_topic)
    loss_emotion = F.cross_entropy(out_emo, true_emo.argmax(dim=1))

    loss = 10 * loss_topic + loss_emotion
    correct_topic = (pred_topic == true_topic).all(dim=1).sum().item()  # 只有完全匹配的才算正确
    correct_emo = (pred_emo == true_emo).all(dim=1).sum().item()  # 只有完全匹配的才算正确
    return loss, correct_topic, correct_emo, pred_topic, pred_emo


def calculate_metrics(true_labels, pred_labels, average='macro'):
    f1 = f1_score(true_labels, pred_labels, average=average)
    precision = precision_score(true_labels, pred_labels, average=average)
    recall = recall_score(true_labels, pred_labels, average=average)
    accuracy = accuracy_score(true_labels, pred_labels)
    return f1, precision, recall, accuracy

def plot_metrics(train_losses, train_topic_accs, train_emo_accs, test_losses, test_topic_accs, test_emo_accs, save_path=None):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 5))

    # 绘制训练和测试的损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, test_losses, 'ro-', label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制主题准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_topic_accs, 'bo-', label='Training Topic Accuracy')
    plt.plot(epochs, test_topic_accs, 'ro-', label='Test Topic Accuracy')
    plt.title('Topic Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制情感准确率曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_emo_accs, 'bo-', label='Training Emotion Accuracy')
    plt.plot(epochs, test_emo_accs, 'ro-', label='Test Emotion Accuracy')
    plt.title('Emotion Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # 如果提供了保存路径，则保存图形
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()
