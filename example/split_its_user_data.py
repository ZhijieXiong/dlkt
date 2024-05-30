import os

import config

from lib.util.data import load_json, write_json


if __name__ == "__main__":
    its_user_path = r"F:\code\myProjects\dlkt\lab\settings\baidu_competition\its_user.json"
    its_user_behavior_path = r"F:\code\myProjects\dlkt\lab\settings\baidu_competition\its_user_behavior.json"
    its_user_portrait_path = r"F:\code\myProjects\dlkt\lab\settings\baidu_competition\its_user_portrait.json"
    target_dir = r"F:\code\myProjects\dlkt\lab\settings\baidu_competition"

    its_users = load_json(its_user_path)
    its_users_behaviors = load_json(its_user_behavior_path)
    its_users_portrait = load_json(its_user_portrait_path)

    num_selected_user = 50
    selected_users = []
    selected_users_behaviors = []
    for i in range(num_selected_user):
        user = its_users[i]
        name = user["name"]
        user_behaviors = list(filter(lambda behavior: behavior["user_name"] == name, its_users_behaviors))

        if name in its_users_portrait.keys() and len(user_behaviors) > 1:
            user_portrait = its_users_portrait[name]
            user["ability"] = user_portrait["ability"]
            user["latent"] = user_portrait["latent"]
            selected_users.append(user)
            selected_users_behaviors.extend(user_behaviors)

    write_json(selected_users, os.path.join(target_dir, "its_user_part.json"))
    write_json(selected_users_behaviors, os.path.join(target_dir, "its_user_behavior_part.json"))
