import copy
import logging
import warnings

import torch
import torch.nn as nn

from .export_utils import optimize_onnx_model, gen_filename
from .export_modules import model_info
from .function import register_custom_ops, domain_info


from .export_modules import (
    ExportQonnxQuantAct,
    ExportQonnxQuantLinear,
    ExportQonnxQuantConv2d,
    ExportQonnxQuantAveragePool2d,
    ExportQonnxQuantBnConv2d,
)
from ..quantization_utils.quant_modules import (
    QuantAct,
    QuantLinear,
    QuantBnConv2d,
    QuantAveragePool2d,
    QuantConv2d,
)

SET_EXPORT_MODE = (
    ExportQonnxQuantAct,
    ExportQonnxQuantLinear,
    ExportQonnxQuantConv2d,
    ExportQonnxQuantAveragePool2d,
    ExportQonnxQuantBnConv2d,
)


# ------------------------------------------------------------
class ExportManager(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        assert model is not None, "Model is not initialized"

        self.copy_model(model)
        self.replace_layers()

    def predict(self, x):
        self.set_export_mode("enable")
        export_pred = self.export_model(x)
        return export_pred

    def forward(self, x):
        self.set_export_mode("disable")
        hawq_pred = self.export_model(x)
        self.set_export_mode("enable")
        export_pred = self.export_model(x)
        return export_pred, hawq_pred

    def copy_model(self, model):
        try:
            self.export_model = copy.deepcopy(model)
        except Exception as e:
            logging.error(e)
            raise Exception(e)

    def replace_layers(self):
        for param in self.export_model.parameters():
            param.requires_grad_(False)

        for name in dir(self.export_model):
            layer = getattr(self.export_model, name)
            onnx_export_layer = None
            if isinstance(layer, QuantAct):
                onnx_export_layer = ExportQonnxQuantAct(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, QuantLinear):
                onnx_export_layer = ExportQonnxQuantLinear(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, QuantConv2d):
                onnx_export_layer = ExportQonnxQuantConv2d(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, QuantBnConv2d):
                onnx_export_layer = ExportQonnxQuantBnConv2d(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, QuantAveragePool2d):
                onnx_export_layer = ExportQonnxQuantAveragePool2d(layer)
                setattr(self.export_model, name, onnx_export_layer)
            elif isinstance(layer, nn.Sequential):
                self.replace_layers(layer)
            # track changes
            if onnx_export_layer is not None:
                model_info["transformed"][layer] = onnx_export_layer
        for param in self.export_model.parameters():
            param.requires_grad_(True)

    @staticmethod
    def enable_export(module):
        if isinstance(module, SET_EXPORT_MODE):
            module.export_mode = True

    @staticmethod
    def disable_export(module):
        if isinstance(module, SET_EXPORT_MODE):
            module.export_mode = False

    def set_export_mode(self, export_mode):
        if export_mode == "enable":
            self.export_model.apply(self.enable_export)
        else:
            self.export_model.apply(self.disable_export)

    def export(self, x, filename=None, save=True):
        assert x is not None, "Input x is not initialized"
        assert type(x) is torch.Tensor, "Input x must be a torch.Tensor"

        if filename is None:
            filename = gen_filename()
        if len(x) > 1:
            logging.info("Only [1, ?] dimensions are supported. Selecting first.")
            x = x[0].view(1, -1)
        register_custom_ops()

        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # collect scaling factors for onnx nodes
                self.set_export_mode("disable")
                _ = self.export_model(x)
                # export with collected values
                self.set_export_mode("enable")
                if save:
                    print("Exporting model...")
                    torch.onnx.export(
                        model=self.export_model,
                        args=x,
                        f=filename,
                        opset_version=11,
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                        custom_opsets={domain_info["name"]: 1},
                    )
                    print("Optimizing...")
                    optimize_onnx_model(filename)
                    print(f"Model saved to: {filename}")
