import pandas as pd
import statsmodels.api as sm

# 求均值和方差
def print_sum_mean(variable_name_zip):
    # 创建一个列表，包含所有的变量
    v_list, name_list = zip(*variable_name_zip)
    # 创建一个空的DataFrame，用于存储结果
    result = []

    # 对每个变量计算平均值和方差，然后添加到结果DataFrame中
    for var, name in zip(v_list, name_list):
        # 均值
        mean = var.mean()
        # 方差
        variance = var.var()
        result.append([name, mean, variance])

    # 将结果列表转换为DataFrame
    result_df = pd.DataFrame(result, columns=['Name', 'Mean', 'Variance'])

    # 输出结果
    print(result_df)

if __name__ == '__main__':
    # 假设你的DataFrame叫df，你可以用pandas的read_excel函数来读取Excel文件
    df = pd.read_excel('followup_studentCN.xlsx')
    name_column = ['w2a0802',
                   'w2a05',
                   'w2b1105', 'w2b1106', 'w2b1107', 'w2b1108', 'w2b1109', 'w2b1110', 'w2b1111',
                   'w2b1201', 'w2b1202', 'w2b1203', 'w2b1204', 'w2b1205', 'w2b1206', 'w2b1207',
                   'w2a09',
                   'w2a15',
                   'w2a16',
                   'w2a17',
                   'w2a22',
                   'w2a23',
                   'w2a2104a',
                   'w2a2104b',
                   'w2a27',
                   'w2a32'
                   ]
    # 删除在class_column列中含有任何缺失值的行
    df = df.dropna(subset=name_column)
    print(df)

    # 计算斯皮尔曼等级相关系数
    # 是否离婚
    marry = df['w2a0802'].map({'是': 1, '否': 0})
    # 是否独生子女
    only_child = df['w2a05'].map({'是': 1, '不是': 0})
    # 辅导班数量
    class_column = ['w2b1105', 'w2b1106', 'w2b1107', 'w2b1108', 'w2b1109', 'w2b1110', 'w2b1111']
    # 计算每一列中'是'的数量，并求和
    # 将每个值与'是'进行比较，如果相等则返回True，否则返回False
    class_sum = df[class_column].applymap(lambda x: x == '是').sum(axis=1)
    # 兴趣爱好数量
    hobby_column = ['w2b1201', 'w2b1202', 'w2b1203', 'w2b1204', 'w2b1205', 'w2b1206', 'w2b1207']
    # 计算每一列中'是'的数量，并求和
    # 将每个值与'是'进行比较，如果相等则返回True，否则返回False
    hobby_sum = df[hobby_column].applymap(lambda x: x == '是').sum(axis=1)
    # 经济条件
    economy = df['w2a09'].map({'非常困难': 1, '比较困难': 2, '中等': 3, '比较富裕': 4, '很富裕': 5})
    # 父亲经常饮酒
    drink = df['w2a15'].map({'是': 1, '否': 0})
    # 父母经常吵架
    quarrel = df['w2a16'].map({'是': 1, '否': 0})
    # 父母关系好不好
    relationship_parent = df['w2a17'].map({'好': 1, '不好': 0})
    # 和父亲关系
    relationship_father = df['w2a22'].map({'不亲近': 1, '一般': 2, '很亲近': 3})
    # 和母亲关系
    relationship_mother = df['w2a23'].map({'不亲近': 1, '一般': 2, '很亲近': 3})
    # 家庭交流(父亲)
    comm_father = df['w2a2104a'].map({'从不': 1, '偶尔': 2, '经常': 3})
    # 家庭交流(母亲)
    comm_mother = df['w2a2104b'].map({'从不': 1, '偶尔': 2, '经常': 3})
    # 对成绩的要求
    grade = df['w2a27'].map({'没有特别要求':1,'班上的平均水平':2,'中上':3,'班上前五名':4})
    # 父母对未来的信息
    confidence = df['w2a32'].map({'根本没有信心':1,'不太有信心':2,'比较有信心':3,'很有信心':4})

    variable_list = [marry,
                     only_child,
                     class_sum,
                     hobby_sum,
                     economy,
                     drink,
                     quarrel,
                     relationship_parent,
                     relationship_father,
                     relationship_mother,
                     comm_father,
                     comm_mother,
                     grade,
                     confidence
                     ]

    name_list = ['marry',
                 'only_child',
                 'class_sum',
                 'hobby_sum',
                 'economy',
                 'drink',
                 'quarrel',
                 'relationship_parent',
                 'relationship_father',
                 'relationship_mother',
                 'comm_father',
                 'comm_mother',
                 'grade',
                 'confidence'
                 ]

    # 求均值和标准差
    variable_name_list = zip(variable_list,name_list)
    # print_sum_mean(variable_name_list)

    # 求线性回归模型
    X = {}
    Y_hobby1 = {}
    Y_class1 = {}

    # 创建自变量DataFrame
    for var,name in variable_name_list:
        if name == 'hobby_sum':
            Y_hobby1[name] = var
        elif name == 'class_sum':
            Y_class1[name] = var
        # elif name !='confidence' and name != 'grade':
        #     X[name] = var
        elif name !='only_child' and name != 'economy':
            X[name] = var
    X =pd.DataFrame(X)
    # 添加截距项
    X = sm.add_constant(X)
    Y_hobby1 = pd.DataFrame(Y_hobby1)
    # 创建模型
    model_hobby1 = sm.OLS(Y_hobby1, X)

    # 拟合模型
    results = model_hobby1.fit()

    # 获取稳健标准差
    robust_results = results.get_robustcov_results()
    # 打印统计结果，包括回归系数、p值等
    print(robust_results.summary())