import torch

def eval_single_dataset(model, dataloader, device):

    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        top1, correct, n, total_loss = 0.0, 0.0, 0.0, 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            x  = inputs.to(device)
            y  = labels.to(device)

            logits = model(x)

            loss = loss_fn(logits, y)

            total_loss += loss.item()

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n
        total_loss /= (i + 1)

    metrics = {"loss": total_loss, "acc": top1}

    return metrics