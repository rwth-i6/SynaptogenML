from mnist_helper import run_training
import pytest
import torch
from torch import nn
from synaptogen_ml.quant_modules import LinearQuant, ActivationQuantizer
from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear


class TilingLinearModel(nn.Module):
    def __init__(self, model_dim=512, num_cycles: int = 0):
        super().__init__()
        self.num_cycles = num_cycles
        self.linear_1 = LinearQuant(
            in_features=28 * 28,
            out_features=model_dim,
            weight_bit_prec=3,
            weight_quant_dtype=torch.qint8,
            weight_quant_method="per_tensor_symmetric",
            bias=True,
        )
        self.final_linear = LinearQuant(
            in_features=model_dim,
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
            output_range_bits=3,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )
        self.memristor_linear_1 = TiledMemristorLinear(
            in_features=28 * 28,
            out_features=model_dim,
            weight_precision=3,
            converter_hardware_settings=hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )
        self.memristor_final = TiledMemristorLinear(
            in_features=model_dim,
            out_features=10,
            weight_precision=3,
            converter_hardware_settings=hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
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
            self.activation_quant_l1_in, self.linear_1, num_cycles_init=self.num_cycles
        )
        self.memristor_final.init_from_linear_quant(
            self.activation_quant_final_in,
            self.final_linear,
            num_cycles_init=self.num_cycles,
        )


@pytest.mark.tiled_linear
def test_linear():
    run_training(
        TilingLinearModel,
        expected_accuracy=0.5,
        batch_size=100,
        num_cycles=1,
        num_epochs=2,
    )
