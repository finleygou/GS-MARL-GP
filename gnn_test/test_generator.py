import numpy as np

a = 3+(
    4+5
    +6
    +7
)

def b():
    yield (
        3,
        4,
        5,
        6
        )


# 创建生成器对象
gen = b()

# 使用next()函数获取生成器中的元组
try:
    value1, value2, value3, value4 = next(gen)
    # 这里不使用print，而是直接将变量赋值给其他变量或进行其他操作
    # 例如，将这些值添加到列表中
    values = [value1, value2, value3, value4]
    print(values)  # 如果需要查看结果，可以打印出来
except StopIteration:
    pass  # 当生成器中没有更多元素时，停止迭代

# 如果你需要在某个地方显示这些值，可以对列表进行操作
# 例如，你可以在交互式环境中直接输入变量名来查看它们的值
# values