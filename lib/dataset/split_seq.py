from ..util import parse


def dataset_truncate2one_seq(data_uniformed, min_seq_len=2, max_seq_len=200, single_concept=True, from_start=True):
    """
    截断数据，取最前面或者最后面一段，不足的在后面补0
    :param data_uniformed:
    :param single_concept:
    :param from_start:
    :param min_seq_len:
    :param max_seq_len:
    :return:
    """
    data_uniformed = list(filter(lambda item: min_seq_len <= item["seq_len"], data_uniformed))
    result = []
    id_keys, seq_keys = parse.get_keys_from_uniform(data_uniformed)
    for item_data in data_uniformed:
        item_data_new = {key: item_data[key] for key in id_keys}
        seq_len = item_data["seq_len"]
        start_index, end_index = 0, seq_len
        if single_concept:
            if seq_len > max_seq_len and from_start:
                end_index = max_seq_len
            if seq_len > max_seq_len and not from_start:
                start_index = end_index - max_seq_len
        else:
            # 各种复杂操作都是为了防止将一道习题截断成两部分
            question_seq = item_data["question_seq"]
            if seq_len > max_seq_len and from_start:
                # 两种情况（假设最长为200）：
                # [..., A, -1, B, ...]，B是第201个，取0:200
                # [..., A, -1, -1, B, ...]，第201个是-1，取A以前的，不取A
                last_j = 0
                for j, q_id in enumerate(question_seq):
                    if j >= max_seq_len:
                        break
                    if q_id != -1:
                        last_j = j
                if question_seq[max_seq_len] == -1:
                    end_index = last_j - 1
                else:
                    end_index = max_seq_len
            if seq_len > max_seq_len and not from_start:
                # 两种情况（假设最长为200）：
                # [..., B, -1, A, ...]，B是第201个，那么就取A以后的
                # [..., x, B, -1, A, ...]，B是第200个，那么就取B以后的
                for j in list(range(seq_len))[::-1]:
                    if (seq_len - j) > max_seq_len:
                        break
                    q_id = question_seq[j]
                    if q_id != -1:
                        start_index = j
        pad_len = max_seq_len - end_index + start_index
        for k in seq_keys:
            item_data_new[k] = item_data[k][start_index:end_index] + [0] * pad_len
        item_data_new["seq_len"] = end_index - start_index
        item_data_new["mask_seq"] = [1] * item_data_new["seq_len"] + \
                                    [0] * (max_seq_len - item_data_new["seq_len"])
        result.append(item_data_new)
    return result


def truncate2multi_seq4single_concept(item_data, seq_keys, id_keys, max_seq_len):
    """
    将一个用户的数据进行常规处理，即截断补零（truncate_and_pad1用于处理单知识点的数据，包括有知识点信息和无知识点信息的）
    :param item_data:
    :param seq_keys:
    :param id_keys:
    :param max_seq_len:
    :return:
    """
    seq_len = item_data["seq_len"]
    result = []
    if seq_len <= max_seq_len:
        item_data_new = {key: item_data[key] for key in id_keys}
        pad_len = max_seq_len - seq_len
        for k in seq_keys:
            item_data_new[k] = item_data[k][0:seq_len] + [0] * pad_len
        item_data_new["mask_seq"] = [1] * seq_len + [0] * pad_len
        result.append(item_data_new)
    else:
        num_segment = item_data["seq_len"] // max_seq_len
        num_segment = num_segment if (item_data["seq_len"] % max_seq_len == 0) else (num_segment + 1)
        for segment in range(num_segment):
            item_data_new = {key: item_data[key] for key in id_keys}
            start_index = max_seq_len * segment
            if segment == item_data["seq_len"] // max_seq_len:
                # the last segment
                pad_len = max_seq_len - (item_data["seq_len"] % max_seq_len)
                for k in seq_keys:
                    item_data_new[k] = item_data[k][start_index:] + [0] * pad_len
                item_data_new["seq_len"] = item_data["seq_len"] % max_seq_len
                item_data_new["mask_seq"] = [1] * (max_seq_len - pad_len) + [0] * pad_len
            else:
                end_index = max_seq_len * (segment + 1)
                for k in seq_keys:
                    item_data_new[k] = item_data[k][start_index:end_index]
                item_data_new["seq_len"] = max_seq_len
                item_data_new["mask_seq"] = [1] * max_seq_len
            result.append(item_data_new)
    return result


def truncate2multi_seq4multi_concept(item_data, seq_keys, id_keys, max_seq_len):
    """
    将一个用户的数据进行常规处理，即截断补零（truncate_and_pad2用于处理多知识点的数据，确保不会将一道题切分成两部分）
    :param item_data: 
    :param seq_keys: 
    :param id_keys: 
    :param max_seq_len: 
    :return: 
    """""
    seq_len = item_data["seq_len"]
    result = []
    if seq_len <= max_seq_len:
        item_data_new = {key: item_data[key] for key in id_keys}
        pad_len = max_seq_len - seq_len
        for k in seq_keys:
            item_data_new[k] = item_data[k][0:seq_len] + [0] * pad_len
        item_data_new["mask_seq"] = [1] * seq_len + [0] * pad_len
        result.append(item_data_new)
    else:
        # 无误差写法，确保不会将一道题切分成两部分
        question_seq = item_data["question_seq"]
        start_index = 0
        while len(question_seq) > max_seq_len:
            item_data_new = {key: item_data[key] for key in id_keys}
            if question_seq[max_seq_len] != -1:
                # [..., A, -1(max_seq_len), B, ...]，B刚好是第max_seq_len + 1个，则从B切断
                end_index = max_seq_len
            else:
                # [..., A, -1(max_seq_len), -1, B, ...]，B在第max_seq_len + 1个后面，则从A切断
                end_index = max_seq_len
                while question_seq[end_index] == -1:
                    end_index -= 1
            this_seq_len = end_index
            for k in seq_keys:
                item_data_new[k] = item_data[k][start_index:start_index + this_seq_len] + \
                                   [0] * (max_seq_len - this_seq_len)
            start_index += this_seq_len
            item_data_new["mask_seq"] = [1] * end_index + [0] * (max_seq_len - this_seq_len)
            item_data_new["seq_len"] = this_seq_len
            result.append(item_data_new)

            question_seq = question_seq[end_index::]

        item_data_new = {key: item_data[key] for key in id_keys}
        pad_len = max_seq_len - len(question_seq)
        for k in seq_keys:
            item_data_new[k] = item_data[k][(seq_len - len(question_seq)):] + [0] * pad_len
        item_data_new["mask_seq"] = [1] * len(question_seq) + [0] * pad_len
        item_data_new["seq_len"] = len(question_seq)
        result.append(item_data_new)

    return result


def dataset_truncate2multi_seq(data_uniformed, min_seq_len=2, max_seq_len=200, single_concept=True):
    """
    截断数据，不足补0，多的当新数据
    :param data_uniformed:
    :param min_seq_len:
    :param max_seq_len:
    :param single_concept:
    :return:
    """
    data_uniformed = list(filter(lambda item: min_seq_len <= item["seq_len"], data_uniformed))
    result = []

    id_keys, seq_keys = parse.get_keys_from_uniform(data_uniformed)
    for item_data in data_uniformed:
        if single_concept:
            # 单知识点，包括有知识点信息和无知识点信息
            item_data_new = truncate2multi_seq4single_concept(item_data, seq_keys, id_keys, max_seq_len)
        else:
            item_data_new = truncate2multi_seq4multi_concept(item_data, seq_keys, id_keys, max_seq_len)
        result += item_data_new

    result = list(filter(lambda item: min_seq_len <= item["seq_len"], result))
    return result
