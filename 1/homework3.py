# HW 1
from math import sqrt
import operator

class Node:
    def __init__(self, node):
        self.id = node
        self.neighbor = {}
    
    def add_neighbor(self, adj, weight=-1):
        self.neighbor[adj] = weight

    def show_weight(self, adj):
        return self.neighbor[adj]

class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_num = 0

    def __iter__(self):
        return iter(self.nodes)

    def add_node(self, node):
        if node not in self.nodes.keys():
            n = Node(node)
            self.nodes[node] = n
            self.node_num += 1
    
    def add_edge(self, u, v, weight):
        if weight == -1:
            return

        if u not in self.nodes.keys():
            self.add_node(u)
            self.node_num += 1
        
        if v not in self.nodes.keys():
            self.add_node(v)
            self.node_num += 1

        self.nodes[u].add_neighbor(self.nodes[v], weight)
        self.nodes[v].add_neighbor(self.nodes[u], weight)

def readInput(file_name):
    lines = []
    with open(file_name, "r+") as f:
        while True:
            line = f.readline().strip().replace("\n", "")
            if not line:
                break
            else:
                lines.append(line)
    f.close()
    return lines

def processInput(data):
    dic = dict()
    dic["mode"] = data.pop(0)
    dic["grids"] = [int(x) for x in data.pop(0).split(" ")]
    dic["start"] = [int(x) for x in data.pop(0).split(" ")]
    dic["target"] = [int(x) for x in data.pop(0).split(" ")]
    dic["chan_num"] = int(data.pop(0))

    channels = []
    for i in data:
        channels.append([int(x) for x in i.split(" ")])
    dic["channels"] = channels
    return dic

def buildGraph(data):
    chan_num = data["chan_num"]
    channels = data["channels"]
    G = Graph()

    if data["mode"] == "BFS":
        weight = lambda x1, y1, x2, y2: abs(abs(x1 - x2) - abs(y1 - y2)) + max(abs(x1 - x2), abs(y1 - y2)) - abs(abs(x1 - x2) - abs(y1 - y2))
    else:
        weight = lambda x1, y1, x2, y2: 10 * (abs(abs(x1 - x2) - abs(y1 - y2))) + 14 *(max(abs(x1 - x2), abs(y1 - y2)) - abs(abs(x1 - x2) - abs(y1 - y2)))

    # for channel in channels:
    for i in range(len(channels)):
        u = str(channels[i][0]) + " " + str(channels[i][1]) + " " + str(channels[i][2])
        v = str(channels[i][3]) + " " + str(channels[i][1]) + " " + str(channels[i][2])
        G.add_node(u)
        G.add_node(v)

        if data["mode"] == "BFS":
            G.add_edge(u, v, 1)
        else:
            G.add_edge(u, v, abs(channels[i][0] - channels[i][3]))

        for j in range(i+1, len(channels)):
            if channels[j] == channels[i]: 
                continue

            if channels[j][0] == channels[i][0]:
                u = str(channels[j][0]) + " " + str(channels[j][1]) + " " + str(channels[j][2])
                v = str(channels[i][0]) + " " + str(channels[i][1]) + " " + str(channels[i][2])

                G.add_node(u)
                G.add_node(v)

                G.add_edge(u, v, weight(channels[i][1], channels[i][2], channels[j][1], channels[j][2]))

            if channels[j][0] == channels[i][3]:
                u = str(channels[j][0]) + " " + str(channels[j][1]) + " " + str(channels[j][2])
                v = str(channels[i][3]) + " " + str(channels[i][1]) + " " + str(channels[i][2])

                G.add_node(u)
                G.add_node(v)

                G.add_edge(u, v, weight(channels[i][1], channels[i][2], channels[j][1], channels[j][2]))

            if channels[j][3] == channels[i][0]:
                u = str(channels[j][3]) + " " + str(channels[j][1]) + " " + str(channels[j][2])
                v = str(channels[i][0]) + " " + str(channels[i][1]) + " " + str(channels[i][2])

                G.add_node(u)
                G.add_node(v)

                G.add_edge(u, v, weight(channels[i][1], channels[i][2], channels[j][1], channels[j][2]))

            if channels[j][3] == channels[i][3]:
                u = str(channels[j][3]) + " " + str(channels[j][1]) + " " + str(channels[j][2])
                v = str(channels[i][3]) + " " + str(channels[i][1]) + " " + str(channels[i][2])

                G.add_node(u)
                G.add_node(v)

                G.add_edge(u, v, weight(channels[i][1], channels[i][2], channels[j][1], channels[j][2]))

    start = str(data["start"][0]) + " " + str(data["start"][1]) + " " + str(data["start"][2])
    target = str(data["target"][0]) + " " + str(data["target"][1]) + " " + str(data["target"][2])

    if start not in G.nodes:
        G.add_node(start)
        for i in G.nodes:
            cs = start.split(" ")
            ci = i.split(" ")
            if i == start:
                pass
            elif ci[0] == cs[0]:
                cs = start.split(" ")
                ci = i.split(" ")
                G.add_edge(start, i, weight(int(cs[1]), int(cs[2]), int(ci[1]), int(ci[2])))

    
    if target not in G.nodes:
        G.add_node(target)
        for i in G.nodes:
            cs = target.split(" ")
            ci = i.split(" ")
            if i == target:
                pass
            elif ci[0] == cs[0]:
                cs = target.split(" ")
                ci = i.split(" ")
                G.add_edge(target, i, weight(int(cs[1]), int(cs[2]), int(ci[1]), int(ci[2])))


    # print(G.nodes)
    # for i in G.nodes:
    #     print (i)
    #     for j in G.nodes[i].neighbor:
    #         print(j.id, end=",")
    #     print("---")
    # print("Finished")

    return G

def gsearch(data, G):
    start = str(data["start"][0]) + " " + str(data["start"][1]) + " " + str(data["start"][2])
    target = str(data["target"][0]) + " " + str(data["target"][1]) + " " + str(data["target"][2])

    if not G.nodes[start].neighbor:
        print("No way from start")
        return "FAIL"
    if not G.nodes[target].neighbor:
        print("No way to target")
        return "FAIL"

    frontier = [[G.nodes[start], 0, 0]]
    explored_set = []
    while frontier != []:
        node = frontier.pop(0)


        # if node[0].id == target:
        #     print ("SUCCESS")
        #     return

        if node[0].id == target:
            res = []
            # print(explored_set)
            while explored_set != []:
                #print (explored_set[-1][0].id, node[2])
                if explored_set[-1][0].id == node[2]:
                    res = [node[0].id + " " + str(node[0].show_weight(explored_set[-1][0]))] + res
                    node = explored_set.pop()
                    # print(node[0].id)
                else:
                    explored_set.pop()
            res = [G.nodes[start].id + " " + "0"] + res
            # print ("SUCCESS")
            return res


        explored_set = explored_set + [node]
        # print("node:" + node)
        expand = node[0].neighbor
        # print (expand)
        for new_node in expand:
            # print(new_node)
            # print (explored_set)
            if not any(new_node in check for check in explored_set):
                if not any(new_node in ff for ff in frontier):
                    g = new_node.show_weight(node[0]) + float(node[1])
                    new = [new_node, g, node[0].id]
                    if data["mode"] == "BFS":
                        frontier.append(new)
                    elif data["mode"] == "UCS":
                        frontier.append(new)
                        # print (frontier)
                    elif data["mode"] == "A*":
                        cnew = new_node.id.split(" ")
                        f = g  + sqrt((data["target"][1] - int(cnew[1]))**2 + (data["target"][2] - int(cnew[2]))**2)
                        new[1] = f
                        frontier.append(new)
                    frontier.sort(key=operator.itemgetter(1))
                else:
                    for i in range(len(frontier)):
                        if new_node in frontier[i]:
                            ct = i 
                    g = new_node.show_weight(node[0]) + float(node[1])
                    cnew = new_node.id.split(" ")
                    f = g + sqrt((data["target"][1] - int(cnew[1]))**2 + (data["target"][2] - int(cnew[2]))**2)
                    if (data["mode"] != "A*"):
                        if (g < float(frontier[ct][1])):
                            frontier[ct][1] = g
                            frontier[ct][2] = node[0].id
                    elif (data["mode"] == "A*"):
                        if (f < float(frontier[ct][1])):
                            frontier[ct][1] = f
                            frontier[ct][2] = node[0].id



        # print("explored_set===============")
        # print (explored_set)
        # print("frontier===============")
        # print (frontier)
    return "FAIL"

def writeOutput(output):
    with open("output.txt", "w+") as f:
        for i in output:
            f.write(i)
            f.write('\n')
    f.close()

def getPath(temp, i, j, mode):
    ii = i.split(" ")
    ii = [int(elem) for elem in ii]

    jj = j.split(" ")
    jj = [int(elem) for elem in jj]

    if ii[0] == jj[0]:
        x = jj[1] - ii[1]
        y = jj[2] - ii[2]
        new = ii
        pub = min(abs(x), abs(y))
        while (x != 0) or (y != 0):
            if pub != 0:
                new[1] = new[1] + x/abs(x)
                new[2] = new[2] + y/abs(y)
                pub -= 1
                x = x - x/abs(x)
                y = y - y/abs(y)
                new.pop()
                if mode == "BFS":
                    new.append(1)
                else:
                    new.append(14)
            elif x == 0:
                new[2] = new[2] + y/abs(y)
                y = y - y/abs(y)
                new.pop()
                if mode == "BFS":
                    new.append(1)
                else:
                    new.append(10)
            elif y == 0:
                new[1] = new[1] + x/abs(x)
                x = x - x/abs(x)
                new.pop()
                if mode == "BFS":
                    new.append(1)
                else:
                    new.append(10)
            tmp = [str(int(x)) for x in new]
            
            temp.append(" ".join(tmp))


def main():
    lines = readInput("input41.txt")
    data = processInput(lines)
    # print(data)
    G = buildGraph(data)
    re = gsearch(data, G)

    if re == "FAIL":
        writeOutput([re])
    else:

        sum = 0
        for i in re:
            cost = int(i.split(" ")[-1])
            sum = sum + cost

        
        # print(re)
        temp = [] 
        if len(re) == 1:
            temp.append(re[0])
        else:
            for i in range(len(re)):
                if i+1 == len(re):
                    break
                if re[i].split(" ")[0] == re[i+1].split(" ")[0]:
                    if temp != []:
                        if re[i] != temp[-1]:
                                temp.append(re[i])  # Append first node
                    else: 
                            temp.append(re[i])  # Append first node

                    getPath(temp, re[i], re[i+1], data["mode"])
                else:
                    temp.append(re[i + 1])

        temp = [str(sum)] + [str(len(temp))] + temp
        # print (temp)
        writeOutput(temp)

if __name__ == "__main__":
    main()