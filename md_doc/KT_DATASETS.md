- [Datasets Intro and Basic Info](#datasets-intro-and-basic-info)
  - [Seq Info](#seq-info)
  - [Side Info](#side-info)
  - [Text Info](#text-info)
  - [Statics Info](#statics-info)
    - [original](#original)
    - [after preprocessing (for single concept datasets or only question datasets)](#after-preprocessing-for-single-concept-datasets-or-only-question-datasets)
    - [after preprocessing (for multi concept datasets)](#after-preprocessing-for-multi-concept-datasets)
- [Datasets Download](#datasets-download)
  - [All Datasets](#all-datasets)
  - [Assist](#assist)
  - [SLP](#slp)
  - [KDD Cup 2010 Education Data Mining Challenge (algebra and bridge2algebra)](#kdd-cup-2010-education-data-mining-challenge-algebra-and-bridge2algebra)
  - [NeurIPS 2020 Education Challenge (edi2020)](#neurips-2020-education-challenge-edi2020)
  - [Ednet](#ednet)
  - [Statics2011](#statics2011)
  - [Slepemapy](#slepemapy)
  - [Slepemapy-Anatomy](#slepemapy-anatomy)
  - [Moocradar](#Moocradar)


# Datasets Intro and Basic Info

## Seq Info

|    dataset name     | question id | concept id | correct | timestamp | use time | age  | hint | attempt | error type |
| :-----------------: | :---------: | :--------: | :-----: | :-------: | :------: | :--: | :--: | :-----: | :--------: |
|     assist2009      |      T      |     T      |    T    |           |          |      |  T   |    T    |            |
|   assist2009-full   |      T      |     T      |    T    |           |          |      |      |         |            |
|     assist2012      |      T      |     T      |    T    |     T     |    T     |      |  T   |    T    |            |
|     assist2015      |      T      |            |    T    |           |          |      |      |         |            |
|     assist2017      |      T      |     T      |    T    |     T     |    T     |      |  T   |    T    |            |
|      ednet-kt1      |      T      |     T      |    T    |     T     |    T     |      |      |         |            |
|    edi2020-task1    |      T      |     T      |    T    |     T     |          |  T   |      |         |            |
|   edi2020-task34    |      T      |     T      |    T    |     T     |          |  T   |      |         |            |
|         poj         |      T      |            |    T    |     T     |          |      |      |         |     T      |
|     algebra2005     |      T      |     T      |    T    |     T     |          |      |      |         |            |
|     algebra2006     |      T      |     T      |    T    |     T     |          |      |      |         |            |
|     algebra2008     |      T      |     T      |    T    |     T     |          |      |      |         |            |
| bridge2algebra2006  |      T      |     T      |    T    |     T     |          |      |      |         |            |
| bridge2algebra2008  |      T      |     T      |    T    |     T     |          |      |      |         |            |
|      junyi2015      |      T      |     T      |    T    |     T     |    T     |      |  T   |    T    |            |
|       xes3g5m       |      T      |     T      |    T    |     T     |          |      |      |         |            |
|     statics2011     |      T      |     T      |    T    |     T     |          |      |      |         |            |
|  slepemapy-anatomy  |      T      |     T      |    T    |     T     |    T     |      |      |         |            |
|       SLP-bio       |      T      |     T      |    T    |           |          |      |      |         |            |
|       SLP-chi       |      T      |     T      |    T    |           |          |      |      |         |            |
|       SLP-eng       |      T      |     T      |    T    |           |          |      |      |         |            |
|       SLP-geo       |      T      |     T      |    T    |           |          |      |      |         |            |
|       SLP-his       |      T      |     T      |    T    |           |          |      |      |         |            |
|       SLP-mat       |      T      |     T      |    T    |           |          |      |      |         |            |
|       SLP-mat       |      T      |     T      |    T    |           |          |      |      |         |            |
| moocradar-C_2287011 |      T      |     T      |    T    |     T     |          |      |      |    T    |            |
| moocradar-C_797404  |      T      |     T      |    T    |     T     |          |      |      |    T    |            |
| moocradar-C_746997  |      T      |     T      |    T    |     T     |          |      |      |    T    |            |

## Side Info

|  dataset name  | school id | country id | gender | premium_pupil | campus |
| :------------: | :-------: | :--------: | :----: | :-----------: | :----: |
|   assist2009   |     T     |            |        |               |        |
|   assist2012   |     T     |            |        |               |        |
|   assist2017   |     T     |            |        |               |        |
| edi2020-task1  |           |            |   T    |       T       |        |
| edi2020-task34 |           |            |   T    |       T       |        |
|   slepemapy-anatomy    |           |     T      |        |               |        |
|    SLP-bio     |     T     |            |   T    |               |   T    |
|    SLP-chi     |     T     |            |   T    |               |   T    |
|    SLP-eng     |     T     |            |   T    |               |   T    |
|    SLP-geo     |     T     |            |   T    |               |   T    |
|    SLP-his     |     T     |            |   T    |               |   T    |
|    SLP-mat     |     T     |            |   T    |               |   T    |

## Text Info

- with text of concept: `xes3g5m` `assist2009` `assist2012` `assist2017`
- with text of quextion:  `xes3g5m` `moocradar`

## Statics Info

### original

- `concepts` of `SLP-his` and `SLP-geo` in original statics is not true, becase there are some `Resource` records (not interaction with question) which are viewed as `concept_id` in the original data
- `questions` of `algebra` and `bridge2algebra`: The steps for processing the questions for these datasets are the same as those of `pykt-toolkit`. This way of handling it will lead to a particularly large number of questions, which needs to be discussed.
- The processing of the `statics2011` dataset is different from previous work. I think there are concepts (hierarchical) in the original dataset, so the dataset contains information of question and concept after processing.

|    dataset name    | intercations | sequences | concepts | questions |
| :----------------: | :----------: | :-------: | :------: | :-------: |
|     assist2009     |    401756    |   4217    |   123    |   26688   |
|  assist2009-full   |   1011079    |   8519    |   347    |   35978   |
|     assist2012     |   6123270    |   46674   |   265    |  179999   |
|     assist2015     |    708631    |   19917   |    --    |    100    |
|     assist2017     |    942816    |   1709    |   102    |   3162    |
|     ednet-kt1      |      --      |    --     |    --    |    --     |
|   edi2020-task1    |   19834813   |  118971   |   282    |   27613   |
|   edi2020-task34   |      --      |    --     |    --    |    --     |
|        poj         |    996240    |   22916   |    --    |   2750    |
|    algebra2005     |    809694    |    574    |   436    |  210710   |
|    algebra2006     |   2270384    |   1338    |   1703   |  580531   |
|    algebra2008     |   8918054    |   3310    |   1828   |  1259272  |
| bridge2algebra2006 |   3679199    |   1146    |   564    |  207856   |
| bridge2algebra2008 |   20012498   |   6043    |   2302   |  566964   |
|     junyi2015      |   25925992   |  247606   |    --    |    --     |
|      xes3g5m       |      --      |    --     |    --    |    --     |
|    statics2011     |    194947    |    333    |    27    |    --     |
|     slepemapy-anatomy      |   1182065    |   18563   |   246    |    --     |
|      SLP-bio       |    291943    |   1941    |    24    |   1061    |
|      SLP-chi       |    80938     |    624    |    32    |    658    |
|      SLP-eng       |    86531     |    366    |    28    |   1089    |
|      SLP-geo       |    151932    |   1135    |    --    |   1021    |
|      SLP-his       |    317813    |   1610    |    --    |   1317    |
|      SLP-mat       |    248674    |   1499    |    45    |   1133    |
|      SLP-phy       |    107289    |    664    |    54    |   1915    |

### after preprocessing (for single concept datasets or only question datasets)

|  dataset name  | intercations | sequences | concepts | questions |
| :------------: | :----------: | :-------: | :------: | :-------: |
|   assist2012   |   2711813    |   29018   |   265    |   53091   |
|   assist2015   |    683801    |   19840   |    --    |    100    |
|   assist2017   |    864713    |   1709    |   101    |   2803    |
| edi2020-task1  |   19834813   |  118971   |   282    |   27613   |
| edi2020-task34 |   1382727    |   4918    |    53    |    948    |
|  statics2011   |    189297    |    333    |    27    |   1223    |
|   slepemapy-anatomy    |   1173566    |   18540   |   246    |   5730    |
|    SLP-bio     |    291800    |   1941    |    23    |   1058    |
|    SLP-chi     |    80888     |    623    |    31    |    637    |
|    SLP-eng     |    86530     |    366    |    28    |   1089    |
|    SLP-geo     |    149780    |   1135    |    47    |   1011    |
|    SLP-his     |    296711    |   1610    |    22    |   1251    |
|    SLP-mat     |    242722    |   1499    |    44    |   1127    |
|    SLP-phy     |    107288    |    664    |    54    |   1915    |
|   junyi2015    |   25925987   |  247606   |    40    |    817    |
|      poj       |    996240    |   22916   |    --    |   2750    |
| moocradar-C_2287011 | 384928 | 2907 | 180 | 181 |
| moocradar-C_797404 | 182148 | 1806 | 253 | 184 |
| moocradar-C_746997 | 100066 | 1577 | 265 | 550 |

### after preprocessing (for multi concept datasets)

- `Interaction` and `concept` have two values. The former is the value in `multi_concept` format, and the latter is the value in `single_concept` format.
- `max c for q` means that one question in the dataset corresponds to at most several concepts.

|      dataset       |     intercations     | sequences |  concepts   | questions | max c for q |
| :----------------: | :------------------: | :-------: | :---------: | :-------: | :---------: |
|     assist2009     |   338001 \| 283105   |   4163    | 123 \| 149  |   17751   |      4      |
|  assist2009-full   |   526655 \| 433090   |   7003    | 151 \| 247  |   13544   |      6      |
|     ednet-kt1      |  1326640 \| 567306   |   5000    | 188 \| 1462 |   11858   |      6      |
|    algebra2005     |   884102 \| 607025   |    574    | 112 \| 436  |  173113   |      5      |
|    algebra2006     |  1852340 \| 1808571  |   1336    | 491 \| 1703 |  549821   |      7      |
|    algebra2008     | 10924109 \| 6442137  |   3292    | 541 \| 1828 |  788006   |      8      |
| bridge2algebra2006 |  1824328 \| 1817476  |   1146    | 493 \| 564  |  129263   |      5      |
| bridge2algebra2008 | 17286061 \| 12350449 |   5995    | 933 \| 2302 |  351171   |      8      |
|      xes3g5m       |  6412912 \| 5549184  |   18066   | 865 \| 1292 |   7651    |      6      |

# Datasets Download

## All Datasets

[bigdata-ustc: EduData](https://github.com/bigdata-ustc/EduData)

## Assist

assist2009: [download](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010)

assist2009-full: [download](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/combined-dataset-2009-10)

assist2012: [download](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with)

assist2015: [download](https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data)

assist2017: [download](https://sites.google.com/view/assistmentsdatamining/dataset?authuser=0)

## SLP

[download](https://aic-fe.bnu.edu.cn/cgzs/kfsj/index.html)

## KDD Cup 2010 Education Data Mining Challenge (algebra and bridge2algebra)

[download](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp)

## NeurIPS 2020 Education Challenge (edi2020)

[download](https://eedi.com/projects/neurips-education-challenge)

## Ednet

[download](https://github.com/riiid/ednet)

## Statics2011

[download](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507)

## Slepemapy

[download](https://www.fi.muni.cz/adaptivelearning/?a=data)

## Slepemapy-Anatomy

[download](http://data.practiceanatomy.com/)

## Moocradar

[download](https://github.com/THU-KEG/MOOC-Radar)