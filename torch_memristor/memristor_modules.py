from torch import nn
import torch
import numpy as np

from .synaptogen import CellArrayCPU, applyVoltage
from .quant_modules import LinearQuant, ActivationQuantizer


def poly_mul(degree: int, polynomials: torch.Tensor, inputs: torch.Tensor):
    """
    Evaluate a polynomial function on all input elements.

    :param degree: Degree of the polynomial
    :param polynomials: polynomial factors of shape [P]
    :param inputs: inputs ("x") to be evaluated in the shape of [..., I]
    :return: [...., I, 1] outputs with additional broadcast axis
    """
    # initialize output with the polynomial constant and already broadcast to the desired return shape
    broadcast_output = torch.broadcast_to(polynomials[0], size=inputs.shape + (1,))  # [..., I, 1]
    # broadcast inputs to the desired output shape
    inputs = inputs.unsqueeze(-1)  # [..., I, 1]
    for i in range(degree - 1):
        coefficient = polynomials[i + 1]  # [1]
        result = torch.mul(inputs ** (i + 1), coefficient)  # [...., I, 1]
        broadcast_output = result + broadcast_output  # [...., I, 1]
    return broadcast_output


class MemristorArray(nn.Module):
    """
    Torch Module for a Memristor Array.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            low_degree: int = 7,
            high_degree: int = 6,
    ):
        super().__init__()

        # the resistance state can be applied to both polynomials beforehand
        self.resistance_weighted_poly_low = nn.Parameter(torch.empty((low_degree,)),
                                                         requires_grad=False)
        self.resistance_weighted_poly_high = nn.Parameter(torch.empty((high_degree,)),
                                                          requires_grad=False)
        self.r = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self.low_degree = low_degree
        self.high_degree = high_degree

        self.BW = 1e-8  # What is this?
        self.kBT = 1.380649e-23 * 300  # what is this?
        self.e = np.exp(1)

        self.in_features = in_features
        self.out_features = out_features

    def init_from_cell_array_output_major(self, cells: CellArrayCPU):
        LLRS = torch.Tensor(cells.params.LLRS)
        # internal computations are easier if we start from lowest polynomial first, so flip
        LLRS = torch.flip(LLRS, dims=[0])
        self.resistance_weighted_poly_low.data = LLRS

        HHRS = torch.Tensor(cells.params.HHRS)
        HHRS = torch.flip(HHRS, dims=[0])
        self.resistance_weighted_poly_high.data = HHRS

        self.r.data = torch.Tensor(cells.r).resize(self.in_features, self.out_features)

    def forward(self, inputs: torch.Tensor):
        result_low = poly_mul(degree=self.low_degree, polynomials=self.resistance_weighted_poly_low, inputs=inputs)
        result_high = poly_mul(degree=self.high_degree, polynomials=self.resistance_weighted_poly_high,
                                  inputs=inputs)
        result_raw = (1 - self.r) * result_low + self.r * result_high  # [....., I, O]

        johnson_noise = 4 * self.kBT * self.BW * torch.abs(result_raw / inputs.unsqueeze(-1))
        shot_noise = 2 * self.e * torch.abs(result_raw) * self.BW
        sigma_total = torch.sqrt(johnson_noise + shot_noise) * 0.0
        noise = torch.randn(result_raw.shape)
        result_noised = result_raw + noise * sigma_total
        return torch.sum(result_noised, dim=-2)
    
class PairedMemristorArrayV2(nn.Module):
    """
    TorchModule for a Memristor array with pairwise subtracted bitlines
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            low_degree: int = 7,
            high_degree: int = 6,
    ):
        super().__init__()

        # the resistance state can be applied to both polynomials beforehand
        self.resistance_weighted_poly_low = nn.Parameter(torch.empty((low_degree,)),
                                                         requires_grad=False)
        self.resistance_weighted_poly_high = nn.Parameter(torch.empty((high_degree,)),
                                                          requires_grad=False)
        self.r = nn.Parameter(torch.empty((in_features, out_features*2)), requires_grad=False)
        self.low_degree = low_degree
        self.high_degree = high_degree

        self.BW = 1e-8  # What is this?
        self.kBT = 1.380649e-23 * 300  # what is this?
        self.e = np.exp(1)

        self.in_features = in_features
        self.out_features = out_features

    def init_from_cell_array_input_major(self, positive_cells: CellArrayCPU, negative_cells: CellArrayCPU):
        assert all(positive_cells.params.LLRS == negative_cells.params.LLRS)
        LLRS = torch.Tensor(positive_cells.params.LLRS)
        # internal computations are easier if we start from lowest polynomial first, so flip
        LLRS = torch.flip(LLRS, dims=[0])
        # extend to [P, O, I]
        #LLRS = torch.reshape(LLRS, shape=[LLRS.shape[0], 1, 1])  # [P, 1, 1]
        #LLRS = torch.expand_copy(LLRS, size=(LLRS.shape[0], 1, self.in_features))  # [P, 1, I]
        # assign to weight
        self.resistance_weighted_poly_low.data = LLRS
        HHRS = torch.Tensor(positive_cells.params.HHRS)
        HHRS = torch.flip(HHRS, dims=[0])
        #HHRS = torch.reshape(HHRS, shape=[HHRS.shape[0], 1, 1])
        #HHRS = torch.expand_copy(HHRS, size=(HHRS.shape[0], 1, self.in_features))
        self.resistance_weighted_poly_high.data = HHRS
        positive_r = torch.Tensor(positive_cells.r).resize(self.in_features, self.out_features)
        negative_r = torch.Tensor(negative_cells.r).resize(self.in_features, self.out_features)
        self.r.data = torch.concat([positive_r, negative_r], dim=-1)  # [I, 2 * O]

    def forward(self, inputs: torch.Tensor):
        # Apply the polynomials to the inputs
        result_low = poly_mul(degree=self.low_degree, polynomials=self.resistance_weighted_poly_low, inputs=inputs)
        result_high = poly_mul(degree=self.high_degree, polynomials=self.resistance_weighted_poly_high,
                                  inputs=inputs)

        # broadcast result to output layer
        result_low = torch.expand_copy(result_low, inputs.shape + (self.out_features*2,))
        result_high = torch.expand_copy(result_high, inputs.shape + (self.out_features*2,))

        # linear interpolation of low and high state with the resistance factors of shape [I, O]
        result_raw = (1 - self.r) * result_low + self.r * result_high  # [....., I, O]

        # compute gaussian noises
        johnson_noise = 4 * self.kBT * self.BW * torch.abs(result_raw / (torch.abs(inputs.unsqueeze(-1)) + 1e-12))
        shot_noise = 2 * self.e * torch.abs(result_raw) * self.BW
        sigma_total = torch.sqrt(johnson_noise + shot_noise)
        noise = torch.randn(result_raw.shape, device=inputs.device)

        # Apply noise
        result_noised = result_raw + noise * sigma_total
        # result_noised = result_raw

        # Sum the output lines
        line_summed = torch.sum(result_noised, dim=-2) # [...., 2*O]
        shape = line_summed.shape

        # Perform the subtraction of positive and negative lines
        axis_extended = torch.reshape(line_summed, shape[:-1] + (2, -1))  # [...., 2, O]
        intermediate = torch.moveaxis(axis_extended, -2, 0)  # [2, ...., O]
        result_noised_paired = intermediate[0] - intermediate[1]  # [...., O]

        return result_noised_paired


from dataclasses import dataclass

@dataclass
class DacAdcHardwareSettings:
    input_bits: int
    output_precision_bits: int
    output_range_bits: int
    hardware_input_vmax: float
    hardware_output_current_scaling: float

    def __post_init__(self):
        self.dac_input_quant_scaling_factor = 1 / 2**(self.input_bits-1)
        self.dac_max = 2**(self.input_bits - 1)
        self.dac_min = -self.dac_max
        self.adc_max = 2**(self.output_precision_bits + self.output_range_bits - 1)
        self.adc_min = -self.adc_max

class DacAdcPair(nn.Module):
    """
    Simple variant of an DAC and ADC Module,
    computationally related to "input quantizer" and "output quantizer"
    """
    def __init__(
        self,
        hardware_settings: DacAdcHardwareSettings
    ):
        super().__init__()
        """
        :param hardware_settings: Should usually be the same for the whole setup / every instance
        """
        self.hs = hardware_settings

    def dac(self, tensor: torch.Tensor):
        int_quantized_input = torch.fake_quantize_per_tensor_affine(
            tensor,
            scale=self.hs.dac_input_quant_scaling_factor,
            zero_point=0,
            quant_min=self.hs.dac_min,
            quant_max=self.hs.dac_max
        )
        # TODO: DAC Noise
        # * self.HARDWARE_VMAX -> convert to physical voltage
        input_voltage = int_quantized_input * self.hs.hardware_input_vmax
        return input_voltage

    def adc(self, tensor: torch.Tensor):
        # go from physical current to internal value space again
        adc_in = tensor * self.hs.hardware_output_current_scaling
        #print("adc_in")
        #print(torch.max(adc_in))
        # the precision should quantize [-1,1], while everything larger should be covered by output bits,
        # so for 5 output bits we are quantizing a range of [-32, 32]
        quantized_output = torch.fake_quantize_per_tensor_affine(
            adc_in,
            scale=1/(2**self.hs.output_precision_bits),
            zero_point=0,
            quant_min=self.hs.adc_min,
            quant_max=self.hs.adc_max,
        )
        return quantized_output


class MemristorLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, weight_precision: int):
        super().__init__()
        self.memristors = torch.nn.ModuleList(
               [
                PairedMemristorArray(in_features, out_features) for _ in range(weight_precision)
            ]
        )

    def init_from_linear_quant(self, activation_quant: ActivationQuantizer, linear_quant: LinearQuant):
        quant_weights = linear_quant.weight_fake_quant(linear_quant.weight)
        quant_weights_scaled = torch.torch.round(quant_weights / linear_quant.weight_fake_quant.scale).to(dtype=torch.int32)
        quant_weights_scaled_transposed = torch.transpose(quant_weights_scaled, 0, 1)  # [out, in] -> [in, out]
        flat = torch.flatten(quant_weights_scaled_transposed)
        positive_weights = torch.clamp(flat, 0, 1).numpy()
        negative_weights = torch.abs(torch.clamp(flat, -1, 0)).numpy()
        size = flat.shape[0]
        positive_cells = CellArrayCPU(size)
        negative_cells = CellArrayCPU(size)
        applyVoltage(positive_cells, positive_weights*-2.0)
        applyVoltage(negative_cells, negative_weights*-2.0)
        print(flat)[:20]
        print(positive_cells.r[:20])
        print(negative_cells.r[:20])
        



def linear_quant_to_mem(linear_quant: LinearQuant, use_v2=False):
    quant_weights = linear_quant.weight_fake_quant(linear_quant.weight)
    quant_weights_scaled = torch.torch.round(quant_weights / linear_quant.weight_fake_quant.scale).to(dtype=torch.int32)
    quant_weights_scaled_transposed = torch.transpose(quant_weights_scaled, 0, 1)  # [out, in] -> [in, out]
    flat = torch.flatten(quant_weights_scaled_transposed).cpu()
    positive_weights = torch.clamp(flat, 0, 1).numpy()
    negative_weights = torch.abs(torch.clamp(flat, -1, 0)).numpy()
    size = flat.shape[0]
    positive_cells = CellArrayCPU(size)
    negative_cells = CellArrayCPU(size)
    applyVoltage(positive_cells, positive_weights * -2.0)
    applyVoltage(negative_cells, negative_weights * -2.0)

    pma = PairedMemristorArrayV2(in_features=linear_quant.in_features, out_features=linear_quant.out_features)
    pma.init_from_cell_array_input_major(positive_cells, negative_cells)

    hardware_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=2,
        output_range_bits=6,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.
    )
    wandler = DacAdcPair(hardware_settings=hardware_settings)
    return pma, wandler


def linear_quant_to_mem_tester(example_input: torch.Tensor, activation_quant: ActivationQuantizer, linear_quant: LinearQuant):
    quant_weights = linear_quant.weight_fake_quant(linear_quant.weight)
    quant_weights_scaled = torch.torch.round(quant_weights / linear_quant.weight_fake_quant.scale).to(dtype=torch.int32)
    quant_weights_scaled_transposed = torch.transpose(quant_weights_scaled, 0, 1)  # [out, in] -> [in, out]
    flat = torch.flatten(quant_weights_scaled_transposed)
    positive_weights = torch.clamp(flat, 0, 1).numpy()
    negative_weights = torch.abs(torch.clamp(flat, -1, 0)).numpy()
    size = flat.shape[0]
    positive_cells = CellArrayCPU(size)
    negative_cells = CellArrayCPU(size)
    applyVoltage(positive_cells, positive_weights * -2.0)
    applyVoltage(negative_cells, negative_weights * -2.0)
    print(flat[:20])
    print(positive_cells.r[:20])
    print(negative_cells.r[:20])

    pma = PairedMemristorArrayV2(in_features=linear_quant.in_features, out_features=linear_quant.out_features)
    pma.init_from_cell_array_input_major(positive_cells, negative_cells)

    scale = activation_quant.scale
    print("scale")
    print(scale)
    print(torch.max(example_input))
    normalized_input = example_input / activation_quant.scale / activation_quant.quant_max
    print(torch.max(normalized_input))
    
    hardware_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=5,
        output_range_bits=6,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.
    )
    wandler = DacAdcPair(hardware_settings=hardware_settings)
    mem_in = wandler.dac(normalized_input)
    print("mem in")
    print(torch.max(mem_in))
    print("mem out")
    mem_out = pma(mem_in)
    print(torch.max(mem_out))
    out = wandler.adc(mem_out)
    print("adc out")
    print(torch.max(out))
    print("weight_max_scale")
    print(linear_quant.weight_fake_quant.scale)
    scaled_out = out*linear_quant.weight_fake_quant.scale * activation_quant.scale * activation_quant.quant_max
    print("value out")
    print(torch.max(scaled_out))

    return scaled_out


def test_toy_memristor():

    # out x in, 3 out, 2 in
    weight_array = np.asarray(
        [
            [1.0, 0.0],
            [1.0, -1.0],
            [0.0, -1.0],
        ]
    )
    quant_weights_scaled = torch.tensor(weight_array).to(dtype=torch.int32)
    quant_weights_scaled_transposed = torch.transpose(quant_weights_scaled, 0, 1)  # [out, in] -> [in, out]
    flat = torch.flatten(quant_weights_scaled_transposed)
    positive_weights = torch.clamp(flat, 0, 1).numpy()
    negative_weights = torch.abs(torch.clamp(flat, -1, 0)).numpy()
    size = flat.shape[0]
    positive_cells = CellArrayCPU(size)
    negative_cells = CellArrayCPU(size)
    applyVoltage(positive_cells, positive_weights * -2.0)
    applyVoltage(negative_cells, negative_weights * -2.0)
    print(flat[:20])
    print(positive_cells.r[:20])
    print(negative_cells.r[:20])

    pma = PairedMemristorArrayV2(in_features=2, out_features=3)
    pma.init_from_cell_array_input_major(positive_cells, negative_cells)

    hardware_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=5,
        output_range_bits=5,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.
    )
    wandler = DacAdcPair(hardware_settings=hardware_settings)
    # B x I = 1x2
    normalized_input = torch.Tensor(
        [[1.0, -0.5]]
    )
    mem_in = wandler.dac(normalized_input)
    print("mem in")
    print(mem_in)
    out = wandler.adc(pma(mem_in))
    print(out)




def test_polymul():
    poly_weight = torch.abs(torch.rand((4)))
    print(poly_weight)
    inp = torch.abs(torch.rand((6, 8, 2)))
    print(inp[0][0])
    result = poly_mul(3, poly_weight, inp)
    print(result[0][0])



def memristor_tests():

    test_polymul()

    torch.set_printoptions(precision=8)

    from .synaptogen import CellArrayCPU, applyVoltage, Iread

    cells = CellArrayCPU(2)  # 1 in 2 out, simple differential
    print(cells.params.HHRSdeg)
    print(cells.params.LLRSdeg)
    print(cells.params.HHRS.shape)
    print(cells.params.LLRS.shape)
    print(cells.r)

    input_high = np.asarray([0.6])
    input_low = np.asarray([0.01])

    torch_cells = MemristorArray(1, 2, cells.params.LLRSdeg, cells.params.HHRSdeg)
    LLRS = torch.Tensor(cells.params.LLRS)
    LLRS = torch.flip(LLRS, dims=[0])
    torch_cells.resistance_weighted_poly_low.data = LLRS
    HHRS = torch.Tensor(cells.params.HHRS)
    HHRS = torch.flip(HHRS, dims=[0])
    torch_cells.resistance_weighted_poly_high.data = HHRS
    torch_cells.r.data = torch.Tensor(cells.r).resize(1, 2).transpose(0, 1)

    from .synaptogen import polyval

    low_poly = polyval(cells.params.LLRS, input_high)
    print(f"low_poly: {low_poly}")
    low_poly_torch = poly_mul(cells.params.LLRSdeg, torch_cells.resistance_weighted_poly_low, torch.Tensor(input_high))
    print(f"low_poly_torch {low_poly_torch}")

    high_poly = polyval(cells.params.HHRS, input_high)
    print(f"high_poly: {high_poly}")
    high_poly_torch = poly_mul(cells.params.HHRSdeg, torch_cells.resistance_weighted_poly_high, torch.Tensor(input_high))
    print(f"high_poly_torch: {high_poly_torch}")

    from .synaptogen import Imix

    mix = Imix(cells.r, input_high, cells.params.HHRS, cells.params.LLRS)
    print(mix)
    mix_torch = torch_cells(torch.Tensor(input_high))
    print(mix_torch)
    # exit(0)
    # check save U
    # estimation_cells = CellArrayCPU(1)
    # applyVoltage(estimation_cells, -2)
    # for u in np.arange(0, 2, 0.1):
    #     applyVoltage(estimation_cells, u)
    #     out = Iread(estimation_cells, 0.6)
    #     print(f"U: {u:.2}, I {out}")
    # => above 0.7 does changes, saturates at 1.5

    # tune scaling
    estimation_cells = CellArrayCPU(200)

    zero_offsets = []
    zero_offsets_nc = []
    for check in np.arange(0.1, 0.7, 0.1):
        out = Iread(estimation_cells, check)
        zero_offset = np.mean(out)
        zero_offsets_nc.append(zero_offset)
        zero_offsets.append(0)
        print(f"check value: {check:.2} gives zero_offset {zero_offset}")

    zero_offset_nc = np.mean(zero_offsets_nc)

    # applyVoltage(estimation_cells, -2.5)
    applyVoltage(estimation_cells, np.asarray([-2.5] * 100 + [0.0] * 100))
    correction_factors = []
    correction_factors_nc = []
    for i, check in enumerate(np.arange(0.1, 0.7, 0.1)):
        cell_out = Iread(estimation_cells, check)
        out = (cell_out[:100] - cell_out[100:])
        out_nc = (cell_out[:100])
        correction_factor = check / np.mean(out - zero_offsets[i])
        correction_factor_nc = check / np.mean(out_nc - zero_offsets_nc[i])
        correction_factors.append(correction_factor)
        correction_factors_nc.append(correction_factor_nc)
        print(f"check value: {check:.2} gives correction factor {correction_factor}")

    applyVoltage(estimation_cells, 2.5)
    for i, check in enumerate(np.arange(0.1, 0.7, 0.1)):
        out = Iread(estimation_cells, check)
        print(f"check value: {check:.2} gives min raw value: {np.min(out)}")

    correction_factor_final = np.mean(correction_factors) / 0.6
    print("correction_factor_final")
    print(correction_factor_final)
    # ~8020
    correction_factor_nc_final = np.mean(correction_factors_nc) / 0.6

    # Do sweeps to determine best value to get a weight of 0.5
    estimation_cells = CellArrayCPU(2000)
    # set low resistance
    applyVoltage(estimation_cells, -2)
    for u_for_05 in np.arange(0, 2, 0.01):
        applyVoltage(estimation_cells, u_for_05)
        for i, check in enumerate(np.arange(0.1, 0.7, 0.1)):
            out = Iread(estimation_cells, 0.6)
            mean_for_readout_u = (np.mean(out) - zero_offsets[i]) * correction_factor_final
        mean = np.mean(mean_for_readout_u)
        if mean <= 0.5:
            break

    print(u_for_05)

    test_values = np.asarray([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    test_cells = CellArrayCPU(2000)
    for test_value in test_values:
        # multiply with 0
        cell_out = Iread(test_cells, test_value * 0.6)
        outs = (cell_out[:1000] - cell_out[1000:]) * correction_factor_final
        # Non corrected
        outs_nc = ((cell_out[:1000]) - zero_offset_nc) * correction_factor_nc_final
        result = np.mean(outs)
        result_nc = np.mean(outs_nc)
        deviation = np.std(outs)
        deviation_nc = np.std(outs_nc)
        max = np.max(outs)
        max_nc = np.max(outs_nc)
        min = np.min(outs)
        min_nc = np.min(outs_nc)
        error = np.mean(outs ** 2)
        error_nc = np.mean(outs_nc ** 2)
        print(
            f"{test_value} * 0 (    corrected) = {result:.3} +/- {deviation:.3}, min: {min:.4}, max: {max:.4}, error: {error}")
        print(
            f"{test_value} * 0 (non-corrected) = {result_nc:.3} +/- {deviation_nc:.3}, min: {min_nc:.4}, max: {max_nc:.4}, error: {error_nc}")

    test_cells = applyVoltage(test_cells, np.asarray([-2.0] * 1000 + [0.0] * 1000))

    torch_cells = MemristorArray(1, 2000)
    torch_cells.init_from_cell_array_output_major(test_cells)

    for test_value in test_values:
        cell_out = Iread(test_cells, test_value * 0.6)
        cell_out_torch = torch_cells.forward(torch.tensor([test_value * 0.6]))
        outs = (cell_out[:1000] - cell_out[1000:]) * correction_factor_final
        outs_torch = ((cell_out_torch[:1000] - cell_out_torch[1000:]) * correction_factor_final).numpy()
        print(f"torch vs normal: {np.std(outs_torch - outs)}")
        # Non corrected
        outs_nc = ((cell_out[:1000]) - zero_offset_nc) * correction_factor_final
        result = np.mean(outs)
        result_nc = np.mean(outs_nc)
        result_torch = np.mean(outs_torch)
        deviation = np.std(outs)
        deviation_nc = np.std(outs_nc)
        deviation_torch = np.std(outs_torch)
        max = np.max(outs)
        max_nc = np.max(outs_nc)
        min = np.min(outs)
        min_nc = np.min(outs_nc)
        error = np.mean((outs - test_value) ** 2)
        error_nc = np.mean((outs_nc - test_value) ** 2)
        print(
            f"{test_value} * 1 (    corrected) = {result:.3} +/- {deviation:.3}, min: {min:.4}, max: {max:.4}, error: {error}")
        print(f"{test_value} * 1 (tch-corrected) = {result_torch:.3} +/- {deviation_torch:.3}")
        print(
            f"{test_value} * 1 (non-corrected) = {result_nc:.3} +/- {deviation_nc:.3}, min: {min_nc:.4}, max: {max_nc:.4}, error: {error_nc}")

    # repeat the test for the pytorch implementation
    # torch_cells = MemristorLinear(100, 20, low_degree=test_cells.params.LLRSdeg, high_degree=test_cells.params.HHRSdeg)
    # torch_cells.init_from_cell_array_output_major(test_cells)
    # torch_cells = torch.compile(torch_cells)

    # print(torch_cells.resistance_weighted_poly_low[1][0][0])
    # print(test_cells.params.LLRS[-2])

    # for test_value in test_values:
    #     torch_in = torch.ones((100,)) * test_value * 0.6
    #     cell_out = torch_cells(torch_in)
    #     outs = (cell_out[:,:10] - cell_out[:,10:]) * correction_factor_final
    #     result = torch.mean(outs)
    #     deviation = torch.std(outs)
    #     print(f"{test_value} * 1 = {result:.3} +/- {deviation:.3}")


    test_cells = applyVoltage(test_cells, np.asarray([u_for_05] * 1000 + [0.0] * 1000))
    for test_value in test_values:
        cell_out = Iread(test_cells, test_value * 0.6)
        outs = (cell_out[:1000] - cell_out[1000:]) * correction_factor_final
        result = np.mean(outs)
        deviation = np.std(outs)
        max = np.max(outs)
        min = np.min(outs)
        print(f"{test_value} * 0.5 = {result:.3} +/- {deviation:.3}, min: {min}, max: {max}")

    test_cells = applyVoltage(test_cells, np.asarray([2.0] * 2000))
    test_cells = applyVoltage(test_cells, np.asarray([-0.0] * 1000 + [-2.0] * 1000))
    for test_value in test_values:
        cell_out = Iread(test_cells, test_value * 0.6)
        outs = (cell_out[:1000] - cell_out[1000:]) * correction_factor_final
        result = np.mean(outs)
        deviation = np.std(outs)
        print(f"{test_value} * -1.0 = {result:.3} +/- {deviation:.3}")

    # print(cells.r)

    # cell_out = cells @ input_high
    # print("1*1 - 1*0",  (cell_out[0] - cell_out[1])*correction_factor_final)
    # cell_out = cells @ input_low
    # print("0*1 - 0*0:",  (cell_out[0] - cell_out[1])*correction_factor_final)

    # applyVoltage(cells, np.asarray([u_for_05, 0]))
    # print(cells.r)

    # cell_out = cells @ input_high
    # print("1*0.5 - 1*0",  (cell_out[0] - cell_out[1])*correction_factor_final)
    # cell_out = cells @ input_low
    # print("0*0.5 - 0*0:",  (cell_out[0] - cell_out[1])*correction_factor_final)

    # try to set weight to 0.5

if __name__ == "__main__":
    test_toy_memristor()
    memristor_tests()
