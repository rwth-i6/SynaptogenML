import copy

import torch
from torch import nn
import time
from torch_memristor.quant_modules import LinearQuant, ActivationQuantizer
import sys


from torch_memristor.memristor_modules import MemristorLinear, DacAdcHardwareSettings
from mnist_helper import create_mnist_dataloaders


class QuantizedModel2(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear_1 = LinearQuant(
            in_features=28 * 28,
            out_features=512,
            weight_bit_prec=3,
            weight_quant_dtype=torch.qint8,
            weight_quant_method="per_tensor_symmetric",
            bias=False,
        )
        self.final_linear = LinearQuant(
            in_features=512,
            out_features=10,
            weight_bit_prec=3,
            weight_quant_dtype=torch.qint8,
            weight_quant_method="per_tensor_symmetric",
            bias=False,
        )

        self.activation_quant_l1_in = ActivationQuantizer(
            bit_precision=8,
            dtype=torch.qint8,
            method="per_tensor_symmetric",
            channel_axis=None,
            moving_avrg=None,
            reduce_range=False,
        )

        self.activation_quant_l1_out = ActivationQuantizer(
            bit_precision=8,
            dtype=torch.qint8,
            method="per_tensor_symmetric",
            channel_axis=None,
            moving_avrg=None,
            reduce_range=False,
        )

        self.activation_quant_final_in = ActivationQuantizer(
            bit_precision=8,
            dtype=torch.qint8,
            method="per_tensor_symmetric",
            channel_axis=None,
            moving_avrg=None,
            reduce_range=False,
        )

        self.activation_quant_final_out = ActivationQuantizer(
            bit_precision=8,
            dtype=torch.qint8,
            method="per_tensor_symmetric",
            channel_axis=None,
            moving_avrg=None,
            reduce_range=False,
        )

        hardware_settings = DacAdcHardwareSettings(
            input_bits=8,
            output_precision_bits=2,
            output_range_bits=6,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )
        self.memristor_linear_1 = MemristorLinear(
            in_features=28 * 28,
            out_features=512,
            weight_precision=3,
            converter_hardware_settings=hardware_settings,
        )
        self.memristor_final = MemristorLinear(
            in_features=512,
            out_features=10,
            weight_precision=3,
            converter_hardware_settings=hardware_settings,
        )

    def forward(self, image, use_memristor=False):
        inp = torch.reshape(image, shape=(-1, 28 * 28))
        if use_memristor:
            linear_out = self.memristor_linear_1(inp)
        else:
            linear_out = self.linear_1(self.activation_quant_l1_in(inp))
        out1 = nn.functional.tanh(self.activation_quant_l1_out(linear_out))
        if use_memristor:
            logits = self.memristor_final(out1)
        else:
            logits = self.final_linear(self.activation_quant_final_in(out1))
        quant_out = self.activation_quant_final_out(logits)
        return quant_out

    def prepare_memristor(self):
        self.memristor_linear_1.init_from_linear_quant(
            self.activation_quant_l1_in, self.linear_1
        )
        self.memristor_final.init_from_linear_quant(
            self.activation_quant_final_in, self.final_linear
        )


def test_linear():
    """

    :return:
    """
    BATCH_SIZE = 10
    NUM_EPOCHS = 5

    dataloader_train, dataloader_test = create_mnist_dataloaders(BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: %s" % device)

    model = QuantizedModel2()
    model.to(device=device)
    optimizer = torch.optim.RAdam(lr=1e-4, params=model.parameters())

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
            if device == "cpu" and num_examples > 2000:
                # do not train so much on CPU
                break
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
            f"train ce: {total_ce / num_examples:.3f} acc: {total_acc / num_examples:.3f}"
        )
        total_ce = 0
        total_acc = 0
        num_examples = 0
        model.eval()
        print("\nstart normal-quant evaluation")
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
            # if num_examples < 100:
            #    print(time.time() - start_tmp)
            # print(f"CE: {ce/BATCH_SIZE:.3f}  ACC: {acc/BATCH_SIZE:.3f}")
        end_float = time.time() - start
        end_float_avg = end_float / num_examples

        print(
            f"test ce: {total_ce / num_examples:.6f}  acc: {total_acc / num_examples:.6f} time: {end_float} per sample: {end_float_avg}"
        )

        model.prepare_memristor()
        model.to(device=device)

        print("\nstart memristor evaluation")
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

        print(
            f"test memristor ce: {total_ce / num_examples:.6f}  acc: {total_acc / num_examples:.6f} time: {end_float} per sample: {end_float_avg}"
        )
