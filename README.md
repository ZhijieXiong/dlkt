# 一、介绍

## 1、数据格式

|                     | multi_concept | single_concept | only_question |
| ------------------- | ------------- | -------------- | ------------- |
| 单知识点数据集      | no            | yes            | no            |
| 多知识点数据集      | yes           | yes            | yes           |
| 无知识点/习题数据集 | no            | no             | yes           |
| 时间信息            | no            | yes            | yes           |

- `multi_concept` 对于多知识点数据集，将一道多知识点习题拆成多个单知识习题
- `single_concept` 对于单知识点数据集，即习题序列和知识点序列一一对应；对于多知识点数据集，将多知识点组合视为新知识点，则数据集转换为单知识点数据集
- `only_question` 只有习题序列，对于无知识点或者无习题数据集，都将其视为习题序列

## 2、数据集信息

- 单知识点数据集：`assist2012` `assist2017` `edi2020` `SLP` `slepemapy`
  - 注意：`edi2020`是层级知识点，且最低一级都是单知识点。只使用最细粒度的知识点，所以算单知识点数据集
- 多知识点数据集：`assist2009` `ednet-kt1`
  - 注意：对于多知识点数据集，如果知识点不是层级的，则会有两种预处理，分别产生`multi_concept`和`single_concept`；多知识点数据集的`only_question`是对应`multi_concept`的，只是没有拆分习题，因为不用知识点序列
- 无知识点/习题数据集：`assist2015` `statics2011`
