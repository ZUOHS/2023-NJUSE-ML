import random

import cv2


def cal(left, right):
    res = (float(left[0]) - float(right[0])) * (float(left[1]) - float(right[1])) * (float(left[2]) - float(right[2]))
    if res >= 0:
        return res
    else:
        return -res


if __name__ == '__main__':
    # 先把原始图像作为数组读入
    k = int(input())
    img_array = cv2.imread('1.png')
    h, w, c = img_array.shape

    # 接下来随机选择k个中心点
    randHeights = random.sample(range(0, h), k)
    randWeights = random.sample(range(0, w), k)

    centers = [[0 for i in range(3)] for j in range(7)]

    for i in range(0, k):
        for j in range(0, 3):
            centers[i][j] = img_array[randHeights[i]][randWeights[i]][j]

    print(centers)

    # Sum记录每个簇的r，g，g之和，以及样本点的数量
    # records记录每个点属于哪一个簇
    Sum = [[0 for i in range(4)] for j in range(7)]
    records = [[0 for i in range(w)] for j in range(h)]

    # 循环十次
    for n in range(10):
        # 清空Sum数组
        for i in range(k):
            for j in range(4):
                Sum[i][j] = 0

        # 遍历每一个点，找到距离最近的簇，进行分类
        for i in range(h):
            for j in range(w):
                Min = 10000000000
                for m in range(k):
                    temp = cal(centers[m], img_array[i][j])
                    if temp < Min:
                        records[i][j] = m
                        Min = temp

                # 维护Sum数组
                for m in range(k):
                    if records[i][j] == m:
                        Sum[m][0] += img_array[i][j][0]
                        Sum[m][1] += img_array[i][j][1]
                        Sum[m][2] += img_array[i][j][2]
                        Sum[m][3] += 1

        # 计算每个簇的平均值，用平均值作为新的中心
        for i in range(k):
            if Sum[i][3] != 0:
                for j in range(3):
                    centers[i][j] = Sum[i][j] / Sum[i][3]
            else:
                temp1 = random.sample(range(0, h), 1)[0]
                temp2 = random.sample(range(0, w), 1)[0]
                for j in range(3):
                    centers[i][j] = img_array[temp1][temp2][j]

    # 用中心点代表簇中的每一个点生成新的图像
    new_array = img_array
    for i in range(h):
        for j in range(w):
            for m in range(3):
                new_array[i][j][m] = centers[records[i][j]][m]

    print(new_array)
    cv2.imwrite('test.jpg', new_array)
