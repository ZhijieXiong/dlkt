import os
import torch


from lib.model.DKT import DKT
from lib.util.data import load_json
from lib.util.basic import str_dict2params


def load_dkt(global_params, global_objects, save_model_dir, ckt_name="saved.ckt", model_name_in_ckt="best_valid"):
    params_path = os.path.join(save_model_dir, "params.json")
    saved_params = load_json(params_path)
    global_params["models_config"] = str_dict2params(saved_params["models_config"])
    global_params["other"] = str_dict2params(saved_params["other"])

    ckt_path = os.path.join(save_model_dir, ckt_name)

    model = DKT(global_params, global_objects).to(global_params["device"])
    if global_params["device"] == "cpu":
        saved_ckt = torch.load(ckt_path, map_location=torch.device('cpu'))
    else:
        saved_ckt = torch.load(ckt_path)
    model.load_state_dict(saved_ckt[model_name_in_ckt])

    return model
