[TOC]

# Datasets Intro and Basic Info

## Seq Info

|    dataset name    | question id | concept id | correct | timestamp | use time | age  |
| :----------------: | :---------: | :--------: | :-----: | :-------: | :------: | :--: |
|     assist2009     |      T      |     T      |    T    |           |          |      |
|  assist2009-full   |      T      |     T      |    T    |           |          |      |
|     assist2012     |      T      |     T      |    T    |     T     |    T     |      |
|     assist2015     |      T      |            |    T    |           |          |      |
|     assist2017     |      T      |     T      |    T    |     T     |    T     |      |
|     ednet-kt1      |      T      |     T      |    T    |     T     |    T     |      |
|   edi2020-task1    |      T      |     T      |    T    |     T     |          |  T   |
|   edi2020-task2    |      T      |     T      |    T    |     T     |          |  T   |
|   edi2020-task34   |      T      |     T      |    T    |     T     |          |  T   |
|      edi2022       |             |            |         |           |          |      |
|    algebra2005     |      T      |     T      |    T    |     T     |          |      |
|    algebra2006     |      T      |     T      |    T    |     T     |          |      |
|    algebra2008     |      T      |     T      |    T    |     T     |          |      |
| bridge2algebra2006 |      T      |     T      |    T    |     T     |          |      |
| bridge2algebra2008 |      T      |     T      |    T    |     T     |          |      |
|     junyi2015      |             |            |         |           |          |      |
|      xes3g5m       |      T      |     T      |    T    |     T     |          |      |
|    statics2011     |      T      |            |    T    |           |          |      |
|     slepemapy      |      T      |     T      |    T    |     T     |          |      |
|      SLP-bio       |      T      |     T      |    T    |           |          |      |
|      SLP-chi       |      T      |     T      |    T    |           |          |      |
|      SLP-eng       |      T      |     T      |    T    |           |          |      |
|      SLP-geo       |      T      |     T      |    T    |           |          |      |
|      SLP-his       |      T      |     T      |    T    |           |          |      |
|      SLP-mat       |      T      |     T      |    T    |           |          |      |
|      SLP-mat       |      T      |     T      |    T    |           |          |      |

## Side Info

|  dataset name  | school id | country id | gender | premium_pupil | campus |
| :------------: | :-------: | :--------: | :----: | :-----------: | :----: |
|   assist2009   |     T     |            |        |               |        |
|   assist2012   |     T     |            |        |               |        |
|   assist2017   |     T     |            |        |               |        |
| edi2020-task1  |           |            |   T    |       T       |        |
| edi2020-task2  |           |            |   T    |       T       |        |
| edi2020-task34 |           |            |   T    |       T       |        |
|   slepemapy    |           |     T      |        |               |        |
|    SLP-bio     |     T     |            |   T    |               |   T    |
|    SLP-chi     |     T     |            |   T    |               |   T    |
|    SLP-eng     |     T     |            |   T    |               |   T    |
|    SLP-geo     |     T     |            |   T    |               |   T    |
|    SLP-his     |     T     |            |   T    |               |   T    |
|    SLP-mat     |     T     |            |   T    |               |   T    |

## Statics Info

### original

|    dataset name    | intercations | sequences | concepts | questions |
| :----------------: | :----------: | :-------: | :------: | :-------: |
|     assist2009     |    401756    |   4217    |   123    |   26688   |
|  assist2009-full   |   1011079    |   8519    |   347    |   35978   |
|     assist2012     |   6123270    |   46674   |   265    |  179999   |
|     assist2015     |              |           |          |           |
|     assist2017     |    942816    |   1709    |   102    |   3162    |
|     ednet-kt1      |      --      |    --     |    --    |    --     |
|   edi2020-task1    |              |           |          |           |
|   edi2020-task2    |              |           |          |           |
|   edi2020-task34   |      --      |    --     |    --    |    --     |
|      edi2022       |              |           |          |           |
|    algebra2005     |    809694    |    574    |   436    |  210710   |
|    algebra2006     |              |           |          |           |
|    algebra2008     |              |           |          |           |
| bridge2algebra2006 |   3679199    |   1146    |   564    |  207856   |
| bridge2algebra2008 |              |           |          |           |
|     junyi2015      |              |           |          |           |
|      xes3g5m       |      --      |    --     |    --    |    --     |
|    statics2011     |              |           |          |           |
|     slepemapy      |              |           |          |           |
|      SLP-bio       |              |           |          |           |
|      SLP-chi       |              |           |          |           |
|      SLP-eng       |              |           |          |           |
|      SLP-geo       |              |           |          |           |
|      SLP-his       |              |           |          |           |
|      SLP-mat       |              |           |          |           |
|      SLP-mat       |              |           |          |           |

### after preprocessing (for single concept datasets or only question datasets)

|  dataset name  | intercations | sequences | concepts | questions |
| :------------: | :----------: | :-------: | :------: | :-------: |
|   assist2012   |   2711813    |   29018   |   265    |   53091   |
|   assist2015   |              |           |          |           |
|   assist2017   |    864713    |   1709    |   101    |   2803    |
| edi2020-task1  |              |           |          |           |
| edi2020-task2  |              |           |          |           |
| edi2020-task34 |   1382727    |   4918    |    53    |    948    |
|  statics2011   |              |           |          |           |
|   slepemapy    |              |           |          |           |
|    SLP-bio     |              |           |          |           |
|    SLP-chi     |              |           |          |           |
|    SLP-eng     |              |           |          |           |
|    SLP-geo     |              |           |          |           |
|    SLP-his     |              |           |          |           |
|    SLP-mat     |              |           |          |           |
|    SLP-mat     |              |           |          |           |

### after preprocessing (for multi concept datasets)

- Please note: `intercations` and `concepts`

|      dataset       |    intercations    | sequences |  concepts   | questions | max c for q |
| :----------------: | :----------------: | :-------: | :---------: | :-------: | :---------: |
|     assist2009     |  338001 \| 283105  |   4163    | 123 \| 149  |   17751   |      4      |
|  assist2009-full   |  526655 \| 433090  |   7003    | 151 \| 247  |   13544   |      6      |
|     ednet-kt1      | 1326640 \| 567306  |   5000    | 188 \| 1462 |   11858   |      6      |
|    algebra2005     |  884102 \| 607025  |    574    | 112 \| 436  |  173113   |      5      |
|    algebra2006     |                    |           |             |           |             |
|    algebra2008     |                    |           |             |           |             |
| bridge2algebra2006 | 1824328 \| 1817476 |   1146    | 493 \| 564  |  129263   |      5      |
| bridge2algebra2008 |                    |           |             |           |             |
|      xes3g5m       | 6412912 \| 5549184 |   18066   | 865 \| 1292 |   7651    |      6      |

# Datasets Download

## All Datasets

[bigdata-ustc: EduData](https://github.com/bigdata-ustc/EduData)

## Assist Datasets

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