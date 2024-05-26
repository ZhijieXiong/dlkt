import os.path

import networkx as nx

import config

from lib.util.data import load_json


if __name__ == "__main__":
    questions_path = "/Users/dream/myProjects/dlkt/lab/dataset_raw/xes3g5m/metadata/questions.json"
    out_put_dir = "/Users/dream/myProjects/dlkt/lab/settings/baidu_competition"
    questions = load_json(questions_path)

    all_kc_routes = []
    root_nodes = set()
    for question in questions.values():
        kc_routes = question["kc_routes"]
        for kc_route in kc_routes:
            kcs = kc_route.split("----")
            kcs = list(map(lambda x: x.strip(), kcs))
            all_kc_routes.append("----".join(kcs))
            root_nodes.add(kcs[0])
    all_kc_routes = list(set(all_kc_routes))
    all_kc_routes = list(map(lambda x: x.split("----"), all_kc_routes))
    # 只用"知识点"
    concepts_kc_routes = list(filter(lambda x: "知识点" in x[0], all_kc_routes))
    concepts_kc_routes = list(map(lambda x: x[1:], concepts_kc_routes))
    # 按照root排序
    concepts_kc_routes = sorted(concepts_kc_routes, key=lambda x: x[0])

    # 存储层次的知识点信息作为RAG的外部库
    with open(os.path.join(out_put_dir, "its_concept_relation.txt"), "w") as f:
        current_root_concept = concepts_kc_routes[0][0]
        for concepts_kc_route in concepts_kc_routes:
            if concepts_kc_route[0] != current_root_concept:
                f.write("#################\n" + "-->".join(concepts_kc_route) + "\n")
                current_root_concept = concepts_kc_route[0]
            else:
                f.write("-->".join(concepts_kc_route) + "\n")

    # 检查是否为DAG
    edges = []
    for concepts_kc_route in concepts_kc_routes:
        for i in range(len(concepts_kc_route) - 1):
            high_level_kc = concepts_kc_route[i]
            low_level_kc = concepts_kc_route[i+1]
            edges.append((high_level_kc, low_level_kc))

    # 创建一个有向图
    G = nx.DiGraph()
    # 向图中添加边
    G.add_edges_from(edges)
    # 检查图是否为有向无环图（DAG）：True
    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"is DAG: {is_dag}")

    # 这个图结构比较复杂
    # knowledge_graph = {}
    # root_nodes = set()
    # for question in questions.values():
    #     kc_routes = question["kc_routes"]
    #     for kc_route in kc_routes:
    #         kcs = kc_route.split("----")
    #         kcs = list(map(lambda x: x.strip(), kcs))
    #         root_nodes.add(kcs[0])
    #
    #         num_kc = len(kcs)
    #         if num_kc == 1:
    #             # 没有
    #             continue
    #         for i in range(num_kc - 1):
    #             high_level_kc = kcs[i]
    #             low_level_kc = kcs[i + 1]
    #             knowledge_graph.setdefault(high_level_kc, set())
    #             knowledge_graph[high_level_kc].add(low_level_kc)
    #
    # all_routes = []
    # for root_node in root_nodes:
    #     # 使用栈来深度遍历
    #     stack4nodes = [root_node]
    #     route_str_list = []
    #     node_has_seen = []
    #
    #     while len(stack4nodes) > 0:
    #         node = stack4nodes.pop()
    #         route_str_list.append(node)
    #         node_has_seen.append(node)
    #
    #         if node in knowledge_graph.keys():
    #             # 非叶子节点
    #             child_nodes = knowledge_graph[node]
    #             for child_node in child_nodes:
    #                 if child_node not in root_nodes and child_node not in node_has_seen:
    #                     stack4nodes.append(child_node)
    #         else:
    #             # 叶子节点
    #             all_routes.append("-->".join(route_str_list))
    #             route_str_list.pop()
    #             node_has_seen.pop()
