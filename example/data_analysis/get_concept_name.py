from lib.data_processor.load_raw import load_csv
import pandas as pd


concept_id_map_path = r"F:\code\myProjects\dlkt\lab/dataset_preprocessed/assist2012/concept_id_map_single_concept.csv"
concept_id2name_map_path = r"F:\code\myProjects\dlkt\lab/dataset_preprocessed/assist2012/concept_id2name_map.csv"

concept_id_map = load_csv(concept_id_map_path)
concept_id2name_map = load_csv(concept_id2name_map_path)
# 合并DataFrame
merged_df = pd.merge(concept_id_map, concept_id2name_map, on='concept_id', how='left')
concept_id2name = merged_df.dropna().set_index('concept_mapped_id')['concept_name'].to_dict()

target_c_id = 1
# 1: Multiplication and Division Integers
# 67,67,67,67,110,110,110,84
# 67: Probability of Two Distinct Events
# 110: Probability of a Single Event
# 84: Area Irregular Figure
print(concept_id2name[target_c_id])
