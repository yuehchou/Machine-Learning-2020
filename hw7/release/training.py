import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):

    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss

def run_epoch(dataloader, teacher_net, student_net, optimizer, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)

        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num
