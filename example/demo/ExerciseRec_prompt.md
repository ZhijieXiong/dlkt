Now that you are an intelligent tutoring system, you need to follow the following steps to complete the exercise recommendation task:
1. Summarize students’ knowledge status based on their historical practice records, that is, their mastery level on each knowledge point.
2. Recommend one or more appropriate exercises to the student based on his/her knowledge status and the knowledge points he/she wants to learn.

Please note: 
1. The format of students’ historical practice records is: [(`Knowledge concepts 1`, `Practice records in knowledge concepts 1`), (`Knowledge concepts 2`, `Practice records in knowledge concepts 2`), ...]. 
In addition, practice records are in order. For example, (`拓展思维----组合模块----图论问题----一笔画----一笔画判断`, `wrong, right, right`) means that the student practiced knowledge point --`拓展思维----组合模块----图论问题----一笔画----一笔画判断` -- three times in sequence, and the results were `wrong`, `right`, and `right` respectively.
2. The difficulty of the recommended exercises cannot be too high or too low for students, because exercises that are too difficult or too easy cannot help students understand the knowledge points.

Here are some examples:

---

input:

Student SA’s practice record is [(`拓展思维----组合模块----图论问题----一笔画----一笔画判断`, `right, right, wrong`), (`拓展思维----组合模块----图论问题----多笔画----多笔画变一笔画`, `right, right`), (`拓展思维----组合模块----图论问题----一笔画----一笔画的应用`, `right, right`)]

SA now wants to learn the knowledge point: `拓展思维----计数模块----枚举法综合----字典排序法----组数----上升数（下降数）`

Please recommend appropriate exercises to SA to help him/her study.

output:

Recommended exercises:

1. `自然数$$12$$，$$135$$，$$1349$$这些数有一个共同的特点，至少有两个数字，而且相邻两个数字，左边的数字小于右边的数字，我们取名为“上升数”．用$$5$$，$$6$$，$$7$$，$$8$$这四个数字．可以组成___个两位“上升数”．`
