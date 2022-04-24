import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import os
import json
def y_up(x):
    return np.sqrt(9*(1-x**2/16) + 1e-10)
def y_down(x):
    return -np.sqrt(9*(1-x**2/16) + 1e-10)
def y_up2(x):
    return np.sqrt(25*(1-x**2/36) + 1e-10)
def y_down2(x):
    return -np.sqrt(25*(1-x**2/36) + 1e-10)
def draw_ellipse():
    """
    绘制椭圆
    """
    x = np.linspace(-4, 4, 150)
    plt.plot(x, y_up(x), "k-", x, y_down(x), "k-")
def acc_dot_draw_circle(X, r):
    """
    以X中position集合为圆心，r为半径作一系列圆

    Args:
        X (_type_): _description_
        r (_type_): _description_
    """
    theta = np.linspace(0, 2*np.pi, 50)
    for center in X:
        X_ = center[0] + r*np.cos(theta)
        Y_ = center[1] + r*np.sin(theta)
        plt.plot(X_, Y_, color="r")
def in_it(dot, search, mode="2d"):
    """
    判断点是否在圆里面

    Args:
        dot (_type_): _description_
        search (_type_): _description_
        mode (str, optional): _description_. Defaults to "2d".

    Returns:
        _type_: _description_
    """
    if mode == "2d":
        return ((dot[0] - search[0])**2 + (dot[1] - search[1])**2) <= search[2]**2
    else:
        return ((dot[0] - search[0])**2 + (dot[1] - search[1])**2 + (dot[2] - search[2])**2) <= search[3]**2
def lap_rate(dot_set, search_in, mode="2d"):
    """
    计算重叠率与孔隙率
    
    Args:
        dot_set (2d/3darray): 点的集合
        search_in (arrary contains of [x, y, [z], r]):圆的集合
        mode (str): dimension. Defaults to "2d".

    Returns:
        lap_rate
    """
    lap_num = 0
    for dot in dot_set:
        is_lap = False
        for search in search_in:
            if in_it(dot, search, mode=mode):
                if is_lap:
                    lap_num += 1
                is_lap = True
    return lap_num / len(dot_set)
def create_dot_set_2d(left, right, y_fn, num_x, mul_y):
    """
    以给定左右边界，均匀产生对应x处边界间二维坐标

    Returns:
        _type_: _description_
    """
    y_up, y_down = y_fn
    x = np.linspace(left, right, num_x)
    x_y = [[x, y] for x in x for y in np.linspace(y_down(x), y_up(x), mul_y*int(abs(y_up(x)-y_down(x))))]
    X = np.array([[0, 0]])
    for i in x_y:
        X = np.concatenate((X, np.array([i])), axis=0)
    return X[1:]
def kmean_alga(X, cluster, seed):
    """
    对X执行聚类算法

    Args:
        X (_type_): _description_

    Returns:
        [postion1, position2...]: 聚类中心的坐标
    """
    kmeans = KMeans(cluster, random_state=seed).fit(X)
    center = kmeans.cluster_centers_
    return center
def del_covered_dot(X, circles, mode="2d"):
    """
    删除X中出现在circles中的点
    
    Returns:
        被删除的点和剩余的点
    """
    remain_dots = []
    del_dots = []
    for x in X:
        will_del = False
        for circle in circles:
            if in_it(x, circle, mode=mode):
                will_del = True
                break
        if will_del:
            del_dots.append(x)
        else:
            remain_dots.append(x)
    return np.array(remain_dots), np.array(del_dots)
def aim(health_cell, cancer_cell, attack_num, attack_radius, seed=1, data=None):
    plt.figure(figsize=(13, 8))
    plt.scatter(health_cell[:, 0], health_cell[:, 1])
    plt.scatter(cancer_cell[:, 0], cancer_cell[:, 1], color="k")
    position = kmean_alga(cancer_cell, attack_num, seed)
    position_r = np.concatenate((position, np.array([[attack_radius]]).repeat(position.shape[0], axis=0)), axis=1)
    acc_dot_draw_circle(position, attack_radius)
    plt.show()
    return health_cell, cancer_cell, position_r
def kill_them(health_cell, cancer_cell, position_r, data=None):
    plt.figure(figsize=(12, 8))
    # draw_ellipse()
    lap_rate_cancer_area = lap_rate(data["cancer_cell_copy"], position_r)
    data["lap_rate"] += lap_rate_cancer_area
    next_cancer_cell, cancer_cell_del = del_covered_dot(cancer_cell, position_r)
    next_health_cell, health_cell_del = del_covered_dot(health_cell, position_r)
    plt.scatter(next_health_cell[:, 0], next_health_cell[:, 1])
    if len(next_cancer_cell) == 0:
        print("All cancer cells was killed!")
    else:
        plt.scatter(next_cancer_cell[:, 0], next_cancer_cell[:, 1], color="k")
        plt.show()
    return next_health_cell, health_cell_del, next_cancer_cell, cancer_cell_del, position_r
def d_not_in(d, mode="2d"):
    if mode == "2d":
        if d[0]**2/16 + d[1]**2/9 > 1:
            return True
    else:
        if d[0]**2 + d[1]**2 + d[2]**2 >16:
            return True
def main_2d():
    i = 1
    input_arg = {}
    print("任意阶段,按回车终止此次程序!")
    on = True
    health_ori = create_dot_set_2d(-6, 6, (y_up2, y_down2), 60, 4)
    health_cell = np.array([d for d in health_ori if d_not_in(d)])
    health_cell_copy = health_cell.copy()

    cancer_cell = create_dot_set_2d(-4, 4, (y_up, y_down), 60, 4)
    cancer_cell_copy = cancer_cell.copy()

    log_data = {"health_cell_copy": health_cell_copy,
                "cancer_cell_copy": cancer_cell_copy,
                "lap_rate": 0,
                "time_cost": 0,
                "position_log": []}
    while on:
        while True:
            while True:
                aim_arguments_str = input("靶向阶段,参数依次为:锁定区数量 锁定区大小 随机种子(可不输入,若输入为整数)") 
                if not aim_arguments_str:
                    on = False
                    break
                aim_arguments_lst = aim_arguments_str.split()
                aim_arguments = [health_cell, cancer_cell, int(aim_arguments_lst[0]), float(aim_arguments_lst[1])]
                if aim_arguments[2] > len(cancer_cell):
                    print("锁定区数目大于癌细胞数目，重新锁定！")
                else:
                    break
            if len(aim_arguments_lst) == 3:
                aim_arguments.append(int(aim_arguments_lst[-1]))
            kill_arguments = aim(*aim_arguments, data=log_data)
            next = int(input("是否重新定靶?(0重新,1继续下一步)"))
            if next:
                break
        if on:
            input_arg[f"No.{i}input"] = aim_arguments_lst
            health_cell, health_cell_del, cancer_cell, cancer_cell_del, position_r = kill_them(*kill_arguments, data=log_data)
            log_data["position_log"].append({"center":position_r[:, 0:2].tolist(), "radius":position_r[0,-1].tolist()})
            print(f"第{i}轮制冷中心(单位皆cm): ", end='')
            for pr in position_r:
                print(f"({pr[0]},{pr[1]})", end="、")
            print(f"\n消杀半径{position_r[0][-1]}", end="\n\n\n")
            i += 1
            if len(cancer_cell) == 0:
                on = False
                plt.show()
                log_data["wrong_kill"] = f"{round(100*(1 - len(health_cell) / len(health_cell_copy)), 2)}%"
                if not os.path.exists('./result'):
                    os.mkdir("./result")
                i = 1
                result_lst = os.listdir("./result")
                while f"No.{i}_2d_result.json" in result_lst:
                    i += 1
                with open(f"./result/No.{i}_2d_result.json", "w") as fp:
                    lr = f"{round(log_data['lap_rate'], 2)} times"
                    will_dump = {"in":input_arg, "out":{
                        "position":log_data["position_log"],
                        "lap_rate":lr,
                        "wrong_kill":log_data["wrong_kill"]}
                               }
                    json.dump(will_dump, fp)
                print(f"\n!!!!!!!lap_rate={lr}, wrong_kill={log_data['wrong_kill']}")


def z_up(x):
    return np.sqrt(4*(1-x[0]**2/16-x[1]**2/9) + 1e-10) + 3
def z_down(x):
    return -np.sqrt(4*(1-x[0]**2/16-x[1]**2/9) + 1e-10) + 3
def z_up2(x):
    return np.sqrt(16-x[0]**2-x[1]**2 + 1e-8)
def z_down2(x):
    return -np.sqrt(16-x[0]**2-x[1]**2 + 1e-8)
def y_up3(x):
    return np.sqrt(16 - x**2 + 1e-10)
def y_down3(x):
    return -np.sqrt(16 - x**2 + 1e-10)
def time_cost(num, r):
    lmd, ro1, k1, c1, ro2, k2, c2 = 37/87, 1000, 920, 0.5, 1.25, 3600, 1600
    return num*r**2/6*(lmd*ro1/k1/c1+(1-lmd)*ro2/k2/c2)
def create_dot_set_3d(left, right, y_fn, z_fn, num_x, mul_y, mul_z):
    X_Y = create_dot_set_2d(left, right, y_fn, num_x, mul_y)
    z_up, z_down = z_fn
    X_Y_Z = [np.concatenate((x_y, np.array([z]))) for x_y in X_Y for z in np.linspace(z_down(x_y) - 1e-10, z_up(x_y) + 1e-10, mul_z*abs(int(z_up(x_y)-z_down(x_y))))]
    return np.array(X_Y_Z)
def acc_dot_draw_circle_3d(X, r, ax):
    """
    以X中position集合为圆心，r为半径作一系列圆

    Args:
        X (_type_): _description_
        r (_type_): _description_
    """
    theta = np.linspace(0, np.pi, 50)
    fi = np.linspace(0, 2*np.pi, 50)
    # print(X, r)
    for center in X:
        X_Y_Z = np.array([[center[0] + r*np.sin(t)*np.cos(f), center[1] + r*np.sin(t)*np.sin(f), center[2] + r*np.cos(t)] for t in theta for f in fi])
        ax.plot3D(X_Y_Z[:, 0], X_Y_Z[:, 1],X_Y_Z[:, 2], 'r')    
def aim_3d(health_cell, cancer_cell, attack_num, attack_radius, seed=1, data=None):
    fig = plt.figure(figsize=(20, 20))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(health_cell[:, 0], health_cell[:, 1], health_cell[:, 2])
    ax.scatter(cancer_cell[:, 0], cancer_cell[:, 1], cancer_cell[:, 2], color="k")
    position = kmean_alga(cancer_cell, attack_num, seed)
    position_r = np.concatenate((position, np.array([[attack_radius]]).repeat(position.shape[0], axis=0)), axis=1)
    acc_dot_draw_circle_3d(position, attack_radius, ax)
    plt.show()
    return health_cell, cancer_cell, position_r
def kill_them_3d(health_cell, cancer_cell, position_r, data=None):
    fig = plt.figure(figsize=(20, 20))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    lap_rate_cancer_area = lap_rate(data["cancer_cell_copy"], position_r)
    data["lap_rate"] += lap_rate_cancer_area
    data["time_cost"] += time_cost(position_r.shape[0], position_r[0, -1])
    next_cancer_cell, cancer_cell_del = del_covered_dot(cancer_cell, position_r, mode="3d")
    next_health_cell, health_cell_del = del_covered_dot(health_cell, position_r, mode="3d")
    ax.scatter(next_health_cell[:, 0], next_health_cell[:, 1], next_health_cell[:, 2])
    if len(next_cancer_cell) == 0:
        print("All cancer cells was killed!")
    else:
        ax.scatter(next_cancer_cell[:, 0], next_cancer_cell[:, 1], next_cancer_cell[:, 2], color="k")
        plt.show()
    return next_health_cell, health_cell_del, next_cancer_cell, cancer_cell_del, position_r
def main_3d():
    input_arg = {}
    i = 1
    print("任意阶段,按回车终止此次程序!")
    on = True
    health_cell = create_dot_set_3d(-4, 4, (y_up3, y_down3), (z_up2, z_down2), 15, 5, 5)
    health_cell_copy = health_cell.copy()
    
    cancer_cell = create_dot_set_3d(-4, 4, (y_up, y_down), (z_up, z_down), 50, 6, 5)
    cancer_cell = np.array([d for d in cancer_cell if d_not_in(d, mode="3d")])
    cancer_cell_copy = cancer_cell.copy()
    log_data = {"health_cell_copy": health_cell_copy,
                "cancer_cell_copy": cancer_cell_copy,
                "lap_rate": 0,
                "time_cost": 0,
                "position_log": []}
    while on:
        while True:
            while True:
                aim_arguments_str = input("靶向阶段,参数依次为:锁定区数量 锁定区大小 随机种子(可不输入,若输入为整数)") 
                if not aim_arguments_str:
                    on = False
                    break
                aim_arguments_lst = aim_arguments_str.split()
                aim_arguments = [health_cell, cancer_cell, int(aim_arguments_lst[0]), float(aim_arguments_lst[1])]
                # print(aim_arguments[2], len(cancer_cell))
                if aim_arguments[2] > len(cancer_cell):
                    print("锁定区数目大于癌细胞数目，重新锁定！")
                else:
                    break
            if on:
                if len(aim_arguments_lst) == 3:
                    aim_arguments.append(int(aim_arguments_lst[-1]))
                kill_arguments = aim_3d(*aim_arguments, data=log_data)
                next = int(input("是否重新定靶?(0重新,1继续下一步)"))
                if next:
                    break
            else:
                break
        if on:
            input_arg[f"No.{i}input"] = aim_arguments_lst
            health_cell, health_cell_del, cancer_cell, cancer_cell_del, position_r = kill_them_3d(*kill_arguments, data=log_data)
            log_data["position_log"].append({"center":position_r[:, 0:3].tolist(), "radius":position_r[0,-1].tolist()})
            print(f"第{i}轮制冷中心(单位皆cm): ", end='')
            for pr in position_r:
                print(f"({pr[0]},{pr[1]},{pr[2]})", end="、")
            print(f"\n消杀半径{position_r[0][-1]}", end="\n\n\n")
            i += 1
            if len(cancer_cell) == 0:
                on = False
                plt.show()
                log_data["wrong_kill"] = f"{round(100*(1 - len(health_cell) / len(health_cell_copy)), 2)}%"
                if not os.path.exists('./result'):
                    os.mkdir("./result")
                i = 1
                result_lst = os.listdir("./result")
                while f"No.{i}_3d_result.json" in result_lst:
                    i += 1
                with open(f"./result/No.{i}_3d_result.json", "w") as fp:
                    lr = f"{round(log_data['lap_rate'], 2)} times"
                    will_dump = {"in":input_arg, "out":{
                        "position":log_data["position_log"],
                        "lap_rate":lr,
                        "wrong_kill":log_data["wrong_kill"],
                        "time_cost":log_data["time_cost"]}
                               }
                    json.dump(will_dump, fp)
                print(f"\n!!!!!!!lap_rate={lr}, wrong_kill={log_data['wrong_kill']}, time_cost={round(log_data['time_cost'], 2)}s")



def executor():
    on = True
    while on:
        j = input("2d or 3d?")
        if j == "2d":
            main_2d()
        elif j == "3d":
            main_3d()
        else:
            print("输入错误")
        j = input("Want to continue?(1 or 0)")
        if not int(j):
            on = False
if __name__ == "__main__":
    executor()