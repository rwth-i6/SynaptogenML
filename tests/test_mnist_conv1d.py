from mnist_helper import run_training
import pytest
import torch
from torch import nn
from synaptogen_ml.memristor_modules import (
    DacAdcHardwareSettings,
    MemristorConv1d,
    TiledMemristorLinear,
)
from synaptogen_ml.quant_modules import ActivationQuantizer, Conv1DQuant, LinearQuant


class ConvModel(nn.Module):
    def __init__(
        self, model_dim: int = 384, kernel_size: int = 15, num_cycles: int = 0
    ):
        super().__init__()

        self.num_cycles = num_cycles
        assert model_dim > 0
        self.model_dim = model_dim

        self.linear_1 = LinearQuant(
            in_features=28,
            out_features=model_dim,
            weight_bit_prec=3,
            weight_quant_dtype=torch.qint8,
            weight_quant_method="per_tensor_symmetric",
            bias=True,
        )
        self.conv_1 = Conv1DQuant(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=kernel_size,
            groups=model_dim,  # TODO check if this causes issues
            stride=1,
            dilation=1,
            padding=(kernel_size - 1) // 2,
            weight_bit_prec=3,
            weight_quant_dtype=torch.qint8,
            weight_quant_method="per_tensor_symmetric",
            bias=True,
        )
        self.final_linear = LinearQuant(
            in_features=28 * model_dim,
            out_features=10,
            weight_bit_prec=3,
            weight_quant_dtype=torch.qint8,
            weight_quant_method="per_tensor_symmetric",
            bias=True,
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

        self.activation_quant_l2_in = ActivationQuantizer(
            bit_precision=8,
            dtype=torch.qint8,
            method="per_tensor_symmetric",
            channel_axis=None,
            moving_avrg=None,
            reduce_range=False,
        )
        self.activation_quant_l2_out = ActivationQuantizer(
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
        self.memristor_linear_1 = TiledMemristorLinear(
            in_features=28,
            out_features=model_dim,
            weight_precision=3,
            converter_hardware_settings=hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )
        self.memristor_conv_1 = MemristorConv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=kernel_size,
            groups=model_dim,
            stride=1,
            padding=(kernel_size - 1) // 2,
            weight_precision=3,
            converter_hardware_settings=hardware_settings,
        )
        self.memristor_final = TiledMemristorLinear(
            in_features=model_dim * 28,
            out_features=10,
            weight_precision=3,
            converter_hardware_settings=hardware_settings,
            memristor_outputs=128,
            memristor_inputs=128,
        )

    def forward(self, image, use_memristor=False):
        inp = torch.reshape(image, shape=(-1, 28, 28))
        if use_memristor:
            # print(inp.shape)
            linear_out = self.memristor_linear_1(inp)
        else:
            linear_out = self.linear_1(self.activation_quant_l1_in(inp))
        out1 = nn.functional.tanh(self.activation_quant_l1_out(linear_out))
        # .reshape(
        #    -1, 8, self.model_dim // 8
        # ))
        if use_memristor:
            # print(out1.shape)
            conv_out = self.memristor_conv_1(out1.transpose(1, 2)).transpose(1, 2)
            # print(f"Memristor {conv_out}")
            # print(f"Non Memristor {self.conv_1(self.activation_quant_l2_in(out1).transpose(1, 2)).transpose(1, 2)}")
            # print(conv_out.shape)
        else:
            conv_out = self.conv_1(
                self.activation_quant_l2_in(out1).transpose(1, 2)
            ).transpose(1, 2)
        out2 = nn.functional.tanh(self.activation_quant_l2_out(conv_out)).reshape(
            -1, self.model_dim * 28
        )
        if use_memristor:
            # print(out2.shape)
            logits = self.memristor_final(out2)
        else:
            logits = self.final_linear(self.activation_quant_final_in(out2))
        quant_out = self.activation_quant_final_out(logits)
        return quant_out

    def prepare_memristor(self):
        self.memristor_linear_1.init_from_linear_quant(
            self.activation_quant_l1_in,
            self.linear_1,
            num_cycles=0,
        )
        self.memristor_conv_1.init_from_conv_quant(
            self.activation_quant_l2_in, self.conv_1, num_cycles=self.num_cycles
        )
        self.memristor_final.init_from_linear_quant(
            self.activation_quant_final_in,
            self.final_linear,
            num_cycles=0,
        )


@pytest.mark.conv
@pytest.mark.linear
def test_conv1d():
    run_training(
        ConvModel, expected_accuracy=0.8, batch_size=100, num_cycles=1, num_epochs=2
    )
    # print("Num Cycle = 1")
    # run_training(ConvModel, expected_accuracy=0.8, batch_size=100, num_cycles=1, num_epochs=1)
    # for x in range(1, 10):
    #     print(f"Num Cycle = {x * 10}")
    #     run_training(ConvModel, expected_accuracy=0.8, batch_size=100, num_cycles=x * 10, num_epochs=1)
