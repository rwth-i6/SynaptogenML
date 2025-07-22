from mnist_helper import run_training
import pytest
from typing import Tuple
import torch
from torch import nn
from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.conv import MemristorConv2d
from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear
from synaptogen_ml.quant_modules import ActivationQuantizer, Conv2dQuant, LinearQuant


class ConvModel(nn.Module):
    def __init__(
        self,
        model_dim: int = 128,
        kernel_size: Tuple[int, int] = (2, 2),
        num_cycles: int = 0,
    ):
        super().__init__()

        self.num_cycles = num_cycles
        assert model_dim > 0
        self.model_dim = model_dim

        self.conv_1 = Conv2dQuant(
            in_channels=1,
            out_channels=model_dim,
            kernel_size=kernel_size,
            groups=1,
            stride=(1, 1),
            dilation=1,
            padding=(1, 1),
            weight_bit_prec=3,
            weight_quant_dtype=torch.qint8,
            weight_quant_method="per_tensor_symmetric",
            bias=True,
        )
        self.conv_2 = Conv2dQuant(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=kernel_size,
            groups=model_dim,
            stride=kernel_size,
            dilation=1,
            padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
            weight_bit_prec=3,
            weight_quant_dtype=torch.qint8,
            weight_quant_method="per_tensor_symmetric",
            bias=True,
        )
        self.final_linear = LinearQuant(
            in_features=28 // (kernel_size[0]) * 28 // (kernel_size[1]) * model_dim,
            out_features=10,
            weight_bit_prec=3,
            weight_quant_dtype=torch.qint8,
            weight_quant_method="per_tensor_symmetric",
            bias=True,
        )

        self.activation_quant_c1_in = ActivationQuantizer(
            bit_precision=8,
            dtype=torch.qint8,
            method="per_tensor_symmetric",
            channel_axis=None,
            moving_avrg=None,
            reduce_range=False,
        )
        self.activation_quant_c1_out = ActivationQuantizer(
            bit_precision=8,
            dtype=torch.qint8,
            method="per_tensor_symmetric",
            channel_axis=None,
            moving_avrg=None,
            reduce_range=False,
        )

        self.activation_quant_c2_in = ActivationQuantizer(
            bit_precision=8,
            dtype=torch.qint8,
            method="per_tensor_symmetric",
            channel_axis=None,
            moving_avrg=None,
            reduce_range=False,
        )
        self.activation_quant_c2_out = ActivationQuantizer(
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
        self.memristor_conv_1 = MemristorConv2d(
            in_channels=1,
            out_channels=model_dim,
            kernel_size=kernel_size,
            groups=1,
            stride=(1, 1),
            padding=(1, 1),
            weight_precision=3,
            converter_hardware_settings=hardware_settings,
        )
        self.memristor_conv_2 = MemristorConv2d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=kernel_size,
            groups=model_dim,
            stride=kernel_size,
            padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
            weight_precision=3,
            converter_hardware_settings=hardware_settings,
        )
        self.memristor_final = TiledMemristorLinear(
            in_features=28 // (kernel_size[0]) * 28 // (kernel_size[1]) * model_dim,
            out_features=10,
            weight_precision=3,
            converter_hardware_settings=hardware_settings,
            memristor_outputs=128,
            memristor_inputs=128,
        )

    def forward(self, image, use_memristor=False):
        inp = torch.reshape(image, shape=(-1, 28, 28))
        inp = inp[:, None, :, :]
        if use_memristor:
            conv1_out = self.memristor_conv_1(inp)
        else:
            conv1_out = self.conv_1(self.activation_quant_c1_in(inp))
        out1 = nn.functional.tanh(self.activation_quant_c1_out(conv1_out))
        if use_memristor:
            conv2_out = self.memristor_conv_2(out1)
        else:
            conv2_out = self.conv_2(self.activation_quant_c2_in(out1))
        out2 = nn.functional.tanh(self.activation_quant_c2_out(conv2_out)).reshape(
            conv2_out.shape[0], -1
        )

        if use_memristor:
            logits = self.memristor_final(out2)
        else:
            logits = self.final_linear(self.activation_quant_final_in(out2))
        quant_out = self.activation_quant_final_out(logits)
        return quant_out

    def prepare_memristor(self):
        self.memristor_conv_1.init_from_conv_quant(
            self.activation_quant_c1_in, self.conv_1, num_cycles_init=self.num_cycles
        )
        self.memristor_conv_2.init_from_conv_quant(
            self.activation_quant_c2_in, self.conv_2, num_cycles_init=self.num_cycles
        )
        self.memristor_final.init_from_linear_quant(
            self.activation_quant_final_in,
            self.final_linear,
            num_cycles_init=0,
        )


@pytest.mark.conv2d
def test_conv1d():
    run_training(
        ConvModel, expected_accuracy=0.5, batch_size=100, num_cycles=1, num_epochs=2
    )
    # print("Num Cycle = 1")
    # run_training(ConvModel, expected_accuracy=0.8, batch_size=100, num_cycles=1, num_epochs=1)
    # for x in range(1, 10):
    #     print(f"Num Cycle = {x * 10}")
    #     run_training(ConvModel, expected_accuracy=0.8, batch_size=100, num_cycles=x * 10, num_epochs=1)
