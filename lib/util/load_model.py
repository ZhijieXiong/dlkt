import os
import torch

from .data import load_json
from .basic import str_dict2params
from ..model.AuxInfoDCT import AuxInfoDCT
from ..model.AuxInfoQDKT import AuxInfoQDKT
from ..model.AKT import AKT
from ..model.AKT_CORE import AKT_CORE
from ..model.ATKT import ATKT
from ..model.AT_DKT import AT_DKT
from ..model.AC_VAE_GRU import AC_VAE_GRU
from ..model.CL4KT import CL4KT
from ..model.DKT import DKT
from ..model.DCT import DCT
from ..model.DIMKT import DIMKT
from ..model.DIMKT_VARIANT import DIMKT_VARIANT
from ..model.DIMKT_VARIANT_CORE import DIMKT_VARIANT_CORE
from ..model.DKVMN import DKVMN
from ..model.DTransformer import DTransformer
from ..model.GIKT import GIKT
from ..model.LBKT import LBKT
from ..model.LPKT import LPKT
from ..model.LPKTPlus import LPKTPlus
from ..model.SAKT import SAKT
from ..model.SAINT import SAINT
from ..model.qDKT import qDKT
from ..model.qDKT_CORE import qDKT_CORE
from ..model.qDKT_CORE_NEW import qDKT_CORE_NEW
from ..model.QIKT import QIKT
from ..model.SimpleKT import SimpleKT
from ..model.MIKT import MIKT
from ..model.SparseKT import SparseKT


model_table = {
    "AuxInfoDCT": AuxInfoDCT,
    "AuxInfoQDKT": AuxInfoQDKT,
    "AKT": AKT,
    "AKT-ADA": AKT,
    "AKT-LfF": AKT,
    "AKT-CORE": AKT_CORE,
    "ATKT": ATKT,
    "AT_DKT": AT_DKT,
    "AC_VAE_GRU": AC_VAE_GRU,
    "CL4KT": CL4KT,
    "DKT": DKT,
    "DCT": DCT,
    "DIMKT": DIMKT,
    "DKVMN": DKVMN,
    "DIMKT-VARIANT": DIMKT_VARIANT,
    "DIMKT-VARIANT-CORE": DIMKT_VARIANT_CORE,
    "DTransformer": DTransformer,
    "GIKT": GIKT,
    "LBKT": LBKT,
    "LPKT": LPKT,
    "LPKTPlus": LPKTPlus,
    "SAKT": SAKT,
    "SAINT": SAINT,
    "qDKT": qDKT,
    "qDKT-ADA": qDKT,
    "qDKT-LfF": qDKT,
    "qDKT-CORE": qDKT_CORE,
    "qDKT-CORE-NEW": qDKT_CORE_NEW,
    "QIKT": QIKT,
    "SimpleKT": SimpleKT,
    "MIKT": MIKT,
    "SparseKT": SparseKT
}


def load_kt_model(global_params, global_objects, save_model_dir, ckt_name="saved.ckt", model_name_in_ckt="best_valid"):
    params_path = os.path.join(save_model_dir, "params.json")
    saved_params = load_json(params_path)
    global_params["models_config"] = str_dict2params(saved_params["models_config"])
    global_params["other"] = str_dict2params(saved_params["other"])
    # global_params["LLM_emb_init"] = str_dict2params(saved_params["LLM_emb_init"])
    global_params["use_LLM_emb4question"] = eval(saved_params.get("use_LLM_emb4question", False))
    global_params["use_LLM_emb4concept"] = eval(saved_params.get("use_LLM_emb4concept", False))

    ckt_path = os.path.join(save_model_dir, ckt_name)
    # 之前kt_model_name是第1个，后面改了命名方式，现在是第0个，为了兼容，用以下方式处理
    kt_model_name0 = os.path.basename(save_model_dir).split("@@")[0]
    kt_model_name1 = os.path.basename(save_model_dir).split("@@")[1]
    if kt_model_name0 in model_table.keys():
        kt_model_name = kt_model_name0
    else:
        kt_model_name = kt_model_name1
    model_class = model_table[kt_model_name]
    if kt_model_name == "LPKT":
        global_objects["LPKT"] = {
            "q_matrix": torch.from_numpy(global_objects["data"]["Q_table"]).float().to(global_params["device"]) + 0.03
        }
        q_matrix = global_objects["LPKT"]["q_matrix"]
        q_matrix[q_matrix > 1] = 1
    if kt_model_name == "LBKT":
        q_gamma = saved_params["models_config"]["kt_model"]["encoder_layer"]["LBKT"]["q_gamma"]
        global_objects["LBKT"] = {
            "q_matrix": torch.from_numpy(global_objects["data"]["Q_table"]).float().to(global_params["device"]) + q_gamma
        }
        q_matrix = global_objects["LBKT"]["q_matrix"]
        q_matrix[q_matrix > 1] = 1
    if kt_model_name == "AuxInfoDCT" or kt_model_name == "AuxInfoQDKT":
        # 聚合时间信息
        global_params["datasets_config"]["test"]["type"] = "kt4lpkt_plus"
        global_params["datasets_config"]["test"]["kt4lpkt_plus"] = {}
    if kt_model_name == "AuxInfoDCT":
        #
        pass
    model = model_class(global_params, global_objects).to(global_params["device"])
    if global_params["device"] == "cpu":
        saved_ckt = torch.load(ckt_path, map_location=torch.device('cpu'))
    else:
        saved_ckt = torch.load(ckt_path)
    model.load_state_dict(saved_ckt[model_name_in_ckt])

    return model
