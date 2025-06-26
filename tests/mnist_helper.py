import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from typing import Type


def create_mnist_dataloaders(batch_size):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = MNIST("./", train=True, download=True, transform=transform)
    dataset2 = MNIST("./", train=False, transform=transform)
    dataloader_train = DataLoader(
        dataset=dataset1,
        batch_size=batch_size,
        shuffle=True,
    )

    dataloader_test = DataLoader(
        dataset=dataset2,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader_train, dataloader_test


def run_training(model: nn.Module, expected_accuracy: float, batch_size: int = 10, num_cycles: int = 0, num_epochs: int = 5):
    from lovely_tensors import monkey_patch

    monkey_patch()

    """

    :return:
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: %s" % device)

    BATCH_SIZE = batch_size
    NUM_EPOCHS = 2 if os.getenv("CI") and device != "cuda" else num_epochs

    dataloader_train, dataloader_test = create_mnist_dataloaders(BATCH_SIZE)

    model = model(num_cycles=num_cycles)
    model.to(device=device)
    optimizer = torch.optim.RAdam(lr=1e-4, params=model.parameters())

    memristor_accs = []

    # do a train step
    for i in range(NUM_EPOCHS):
        print("\nstart train epoch %i" % i)
        total_ce = 0
        total_acc = 0
        num_examples = 0
        model.to(device=device)
        model.train()

        for data in dataloader_train:
            image, labels = data
            num_examples += image.shape[0]
            # if device == "cpu" and num_examples > 2000:
            #     # do not train so much on CPU
            #     break
            image = image.to(device=device)
            labels = labels.to(device=device)
            logits = model.forward(image)
            ce = nn.functional.cross_entropy(logits, target=labels, reduction="sum")
            total_ce += ce.detach().cpu()
            acc = torch.sum(torch.eq(torch.argmax(logits, dim=-1), labels).int())
            total_acc += acc.detach().cpu()
            # print(f"CE: {ce/BATCH_SIZE:.3f}  ACC: {acc/BATCH_SIZE:.3f}")
            ce.backward()

            optimizer.step()
            optimizer.zero_grad()

        print(
            f"Epoch {i+1}: train ce: {total_ce / num_examples:.3f} acc: {total_acc / num_examples:.3f}"
        )
        total_ce = 0
        total_acc = 0
        num_examples = 0
        model.eval()
        #print("\nstart normal-quant evaluation")
        start = time.time()
        for data in dataloader_test:
            start_tmp = time.time()
            image, labels = data
            image = image.to(device=device)
            labels = labels.to(device=device)
            num_examples += image.shape[0]
            with torch.no_grad():
                logits = model.forward(image)
            ce = nn.functional.cross_entropy(logits, target=labels, reduction="sum")
            total_ce += ce.detach().cpu()
            acc = torch.sum(torch.eq(torch.argmax(logits, dim=-1), labels).int())
            total_acc += acc.detach().cpu()

        end_float = time.time() - start
        end_float_avg = end_float / num_examples

        print(
            f"Epoch {i+1}: Normal-quant test ce: {total_ce / num_examples:.6f}, acc: {total_acc / num_examples:.6f}, time: {end_float:.2f}s, per sample: {end_float_avg:.2f}s"
        )

        model.prepare_memristor()
        model.to(device=device)

        #print("\nstart memristor evaluation")
        start = time.time()
        for data in dataloader_test:
            start_tmp = time.time()
            image, labels = data
            image = image.to(device=device)
            labels = labels.to(device=device)
            num_examples += image.shape[0]
            with torch.no_grad():
                logits = model.forward(image, use_memristor=True)
            ce = nn.functional.cross_entropy(logits, target=labels, reduction="sum")
            total_ce += ce.detach().cpu()
            acc = torch.sum(torch.eq(torch.argmax(logits, dim=-1), labels).int())
            total_acc += acc.detach().cpu()
            # if num_examples < 100:
            #    print(time.time() - start_tmp)
            # print(f"CE: {ce/BATCH_SIZE:.3f}  ACC: {acc/BATCH_SIZE:.3f}")
        end_float = time.time() - start
        end_float_avg = end_float / num_examples

        memristor_acc = total_acc / num_examples
        memristor_accs.append(memristor_acc)
        print(
            f"Epoch {i+1}: test memristor ce: {total_ce / num_examples:.6f}, acc: {memristor_acc:.6f}, time: {end_float:.2f}s, per sample: {end_float_avg:.2f}s"
        )

    assert any(
        acc >= expected_accuracy for acc in memristor_accs
    ), f"accuracy too low: {max(memristor_accs):.2f} <= {expected_accuracy:.2f}"
