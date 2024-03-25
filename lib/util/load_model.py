import os
import torch

from .data import load_json
from .basic import str_dict2params
from ..model.AuxInfoDCT import AuxInfoDCT
from ..model.AuxInfoQDKT import AuxInfoQDKT
from ..model.AKT import AKT
from ..model.ATKT import ATKT
from ..model.AT_DKT import AT_DKT
from ..model.AC_VAE_GRU import AC_VAE_GRU
from ..model.CL4KT import CL4KT
from ..model.DKT import DKT
from ..model.DCT import DCT
from ..model.DIMKT import DIMKT
from ..model.DKVMN import DKVMN
from ..model.DTransformer import DTransformer
from ..model.GIKT import GIKT
from ..model.NCD import NCD
from ..model.LBKT import LBKT
from ..model.LPKT import LPKT
from ..model.LPKTPlus import LPKTPlus
from ..model.SAKT import SAKT
from ..model.SAINT import SAINT
from ..model.qDKT import qDKT
from ..model.QIKT import QIKT


model_table = {
    "AuxInfoDCT": AuxInfoDCT,
    "AuxInfoQDKT": AuxInfoQDKT,
    "AKT": AKT,
    "ATKT": ATKT,
    "AT_DKT": AT_DKT,
    "AC_VAE_GRU": AC_VAE_GRU,
    "CL4KT": CL4KT,
    "DKT": DKT,
    "DCT": DCT,
    "DIMKT": DIMKT,
    "DKVMN": DKVMN,
    "DTransformer": DTransformer,
    "GIKT": GIKT,
    "NCD": NCD,
    "LBKT": LBKT,
    "LPKT": LPKT,
    "LPKTPlus": LPKTPlus,
    "SAKT": SAKT,
    "SAINT": SAINT,
    "qDKT": qDKT,
    "QIKT": QIKT,
}


def load_kt_model(global_params, global_objects, save_model_dir, ckt_name="saved.ckt", model_name_in_ckt="best_valid"):
    params_path = os.path.join(save_model_dir, "params.json")
    saved_params = load_json(params_path)
    global_params["models_config"] = str_dict2params(saved_params["models_config"])
    global_params["other"] = str_dict2params(saved_params["other"])

    ckt_path = os.path.join(save_model_dir, ckt_name)
    kt_model_name = os.path.basename(save_model_dir).split("@@")[1]
    model_class = model_table[kt_model_name]
    model = model_class(global_params, global_objects).to(global_params["device"])
    saved_ckt = torch.load(ckt_path)
    model.load_state_dict(saved_ckt[model_name_in_ckt])

    return model
