import numpy
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from numpy import argmax
import networkx as nx
import math
import random
from collections import defaultdict



def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX = trainX / 255.0
    testX = testX / 255.0
    speedIMAGES = numpy.array(testX[5000:])
    speedLABELS = testY[5000:]
    testY = testY[:5000]
    testX = testX[:5000]
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY, speedLABELS, speedIMAGES


def plot_loss_accuracy(history):
    plt.subplot(2, 1, 1)
    plt.title('Loss')
    plt.plot(history.history['loss'], color='coral', label='train')
    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color='green', label='train')
    plt.show()


trainX, trainY, testX, testY, speedLABELS, speedIMAGES = load_dataset()

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(trainX, trainY, epochs=3, batch_size=32, validation_data=(testX, testY), verbose=1)
model.evaluate(testX, testY, verbose=1)
plot_loss_accuracy(history)

predict_value = model.predict(numpy.array([speedIMAGES[11], ]))
print('Predicted Number: ', argmax(predict_value, 1)[0])
print('Actual Number: ', speedLABELS[11])
plt.imshow(numpy.array(speedIMAGES[11]).reshape(28, 28))
plt.show()




# road car agent


class Road:
    def __init__(self, crossroads):
        self.crossroads = crossroads
        num_nodes_side = math.sqrt(crossroads)
        if not num_nodes_side.is_integer():
            raise ValueError("Crossroads parameter should be a perfect square.")
        self.edgeNodes = math.ceil(num_nodes_side)
        self.map = nx.Graph()
        self.crossroadsPositions = {}
        self.roadSpeeds = dict()
        self.colors = []
        for i in range(crossroads):
            self.colors.append("lightgreen")
            self.crossroadsPositions[i] = [i % num_nodes_side, -math.floor(i / num_nodes_side)]
            if (i + 1) % num_nodes_side != 0 and i + 1 < crossroads:
                val = -1
                while val == -1 or int(speedLABELS[val]) < 2:
                    val = random.randint(0, len(speedIMAGES) - 1)
                self.map.add_edge(i, int(i + 1), speed=f'{speedLABELS[val]*10}')
                self.roadSpeeds.update({(i, i + 1): speedIMAGES[val]})
            if (i + num_nodes_side) < crossroads:
                val = -1
                while val == -1 or int(speedLABELS[val]) < 2:
                    val = random.randint(0, len(speedIMAGES) - 1)
                self.map.add_edge(i, int(i + num_nodes_side), speed=f'{speedLABELS[val]*10}')
                self.roadSpeeds.update({(i, i + num_nodes_side): speedIMAGES[val]})

    def draw_road(self):
        nx.draw(
            self.map,
            pos=self.crossroadsPositions,
            with_labels=True,
            node_size=300,
            node_color=self.colors,
        )
        road_labels = nx.get_edge_attributes(self.map, 'speed')
        nx.draw_networkx_edge_labels(self.map, pos=self.crossroadsPositions, edge_labels=road_labels, label_pos=0.5,
                                     font_color='black', font_size=16)
        plt.show()

    def erase_edges(self, e):
        if e > (self.map.number_of_edges() - self.map.number_of_nodes() + 1):
            raise ValueError("Number of edges to erase exceeds the maximum amount for this graph to stay connected.")
        erased = 0
        while erased < e:
            try_edge = list(self.map.edges(data=True))[random.randint(0, len(self.map.edges) - 1)]
            speed = try_edge[2]['speed']
            self.map.remove_edge(try_edge[0], try_edge[1])
            speedImage = self.roadSpeeds.get((try_edge[0], try_edge[1]))
            self.roadSpeeds.pop((try_edge[0], try_edge[1]), None)
            if nx.is_connected(self.map):
                erased += 1
            else:
                self.map.add_edge(try_edge[0], try_edge[1], speed=speed)
                self.roadSpeeds.update({(try_edge[0], try_edge[1]): speedImage})

    def get_node_by_coords(self, x, y):
        return y * self.edgeNodes + x

    def get_coords_by_node(self, node):
        return [node % self.edgeNodes, math.floor(node / self.edgeNodes)]

    def get_adj_edges_of_node(self, x, y, second=False):
        node = self.get_node_by_coords(x, y)
        adj = list(self.map.edges(node))
        n = len(adj)
        edges = {}
        if second:
            edges = []
        for i in range(n):
            nnode = adj[i][1]
            x, y = self.get_coords_by_node(nnode)
            key = '-'.join([str(elem) for elem in [x, y]])
            if second:
                edges.append([x, y])
            else:
                res_second = self.get_adj_edges_of_node(x, y, True)
                speed = self.roadSpeeds.get((node, nnode))
                if speed is None:
                    speed = self.roadSpeeds.get((nnode, node))
                edges[key] = dict(roads=res_second, speed=speed)
        return edges

    def show_move(self, x, y, visited_nodes):
        cur_node = self.get_node_by_coords(x, y)
        visited = []
        for node in visited_nodes:
            x, y = node
            visited.append(self.get_node_by_coords(x, y))
        k = 0
        for node in self.map:
            if node == cur_node:
                self.colors[k] = 'yellow'
            elif node in visited:
                self.colors[k] = 'darkgreen'
            else:
                self.colors[k] = 'lightgreen'
            k += 1


class Path:
    def __init__(self):
        self.path = defaultdict(list)

    def add_road(self, u, v):
        self.path[u].append(v)

    def dfs(self, node, visited, found, dest):
        visited.add(node)
        found.append(node)
        if node == dest:
            return found
        for neighbour in self.path[node]:
            if neighbour not in visited:
                res = self.dfs(neighbour, visited, found, dest)
                if res:
                    return res
                index = found.index(node)
                found = found[:index + 1]

    def search(self, node, dest):
        visited, found = set(), []
        return self.dfs(node, visited, found, dest)


def get_id_from_coords(x, y):
    return str(x) + "-" + str(y)


def get_coords_from_id(key):
    x, y = key.split("-")
    return int(x), int(y)


class KnowledgeBase:
    def __init__(self):
        self.path = Path()
        self.map = nx.Graph()
        self.crossroads_pos = {}
        self.crossroads_data = {}
        self.colors = []
        self.roadSpeeds = {}

    def set_crossroads_data(self, x, y, x1, y1, speed):
        key2 = get_id_from_coords(x, y)
        key1 = get_id_from_coords(x1, y1)
        x = int(x)
        y = int(y)
        x1 = int(x1)
        y1 = int(y1)
        speed_key = key2 + '_' + key1
        self.roadSpeeds[speed_key] = speed
        self.path.add_road(key1, key2)
        if x1 + 1 == x:
            self.crossroads_data[key1][0] = key2
        elif x1 - 1 == x:
            self.crossroads_data[key1][2] = key2
        elif y1 + 1 == y:
            self.crossroads_data[key1][3] = key2
        elif y1 - 1 == y:
            self.crossroads_data[key1][1] = key2

    def set_inner_colors(self, node, explored):
        if node not in self.crossroads_pos:
            if node in explored:
                self.crossroads_data[node] = [0] * 5
                self.colors.append('gold')
            else:
                self.crossroads_data[node] = [0] * 5
                self.colors.append('tomato')
        else:
            self.crossroads_data[node] = [0 if x is None else x for x in self.crossroads_data[node]]
            if node in explored:
                index = list(self.map.nodes).index(node)
                self.colors[index] = 'gold'

    def tell(self, data):
        roads = data['roads']
        speeds = data['speeds']
        explored = data['explored']
        self.roadSpeeds.update(speeds)
        for i in range(len(roads)):
            point1, point2 = roads[i]
            x1, y1 = point1
            x2, y2 = point2
            key1 = get_id_from_coords(x1, y1)
            key2 = get_id_from_coords(x2, y2)
            speed = "?"
            if key1 + "_" + key2 in self.roadSpeeds.keys():
                speed = self.roadSpeeds[key1 + "_" + key2]
            self.set_inner_colors(key1, explored)
            self.set_inner_colors(key2, explored)
            if key2 not in explored and self.crossroads_data[key1][4] == 0:
                self.crossroads_data[key1][4] += 2
            elif key1 not in explored and self.crossroads_data[key2][4] == 0:
                self.crossroads_data[key2][4] += 2
            self.crossroads_pos[key1] = [x1, -y1]
            self.crossroads_pos[key2] = [x2, -y2]
            self.map.add_edge(key1, key2, speed=speed)
            self.set_crossroads_data(x1, y1, x2, y2, speed)
            self.set_crossroads_data(x2, y2, x1, y1, speed)

    def draw_inner(self):
        nx.draw(
            self.map,
            node_color=self.colors,
            pos=self.crossroads_pos,
            with_labels=True,
            node_size=300
        )
        road_labels = nx.get_edge_attributes(self.map, 'speed')
        nx.draw_networkx_edge_labels(self.map, pos=self.crossroads_pos, edge_labels=road_labels, label_pos=0.5,
                                     font_color='black', font_size=16)
        plt.show()

    def ask(self, data):
        curr_x, curr_y = data['current']
        dest_x, dest_y = data['destination']
        curr_key = get_id_from_coords(curr_x, curr_y)
        dest_key = get_id_from_coords(dest_x, dest_y)
        self.crossroads_data[curr_key][4] = -10
        current_node_roads = self.crossroads_data[curr_key]

        if dest_key in self.crossroads_data:
            arr = self.path.search(curr_key, dest_key)
            node = arr[1]
            index = current_node_roads.index(node)
            next_key = current_node_roads[index]
            speed = "?"
            if curr_key + "_" + next_key in self.roadSpeeds.keys():
                speed = self.roadSpeeds[curr_key + "_" + next_key]
            return index, speed

        scores = [-1000] * 4
        deadends = []
        i = 0
        for node in current_node_roads:
            key = i
            i += 1
            if i == 5:
                break
            if node == 0:
                continue
            roads_number = 0
            scores[key] = 0
            is_deadend = False
            if node in self.crossroads_data:
                node_data = self.crossroads_data[node]
                roads_number = 4 - node_data[:4].count(0)
                scores[key] = self.crossroads_data[node][4]
                if scores[key] < -100:
                    is_deadend = True
            deadends.append(roads_number == 1 or is_deadend)
            if roads_number == 1:
                scores[key] += -100
                self.crossroads_data[node][4] += -100
            x, y = get_coords_from_id(node)
            if x == dest_x and y == dest_y:
                scores[key] += 100
            if abs(x - dest_x) < abs(curr_x - dest_x) or abs(y - dest_y) < abs(curr_y - dest_y):
                scores[key] += 5
        if deadends.count(False) == 1:
            self.crossroads_data[curr_key][4] += -100
        elif deadends.count(False) == 0:
            return -1, "?"
        max_value = max(scores)
        max_index = scores.index(max_value)
        speed = "?"
        next_key = current_node_roads[max_index]
        if curr_key + "_" + next_key in self.roadSpeeds.keys():
            speed = self.roadSpeeds[curr_key + "_" + next_key]
        return max_index, speed


class Agent:
    def __init__(self, road, x, y):
        self.visited_nodes = None
        self.x = x
        self.y = y
        self.angle = 0
        self.score = 0
        self.map = road
        self.crossroads = {}
        self.path = Path()
        self.history = [[get_id_from_coords(self.x, self.y), "0"]]
        self.knowledge = KnowledgeBase()

    def check_angle(self):
        while not (0 <= self.angle < 360):
            if self.angle >= 360:
                self.angle -= 360
            elif self.angle < 0:
                self.angle += 360

    def left(self):
        print("Turn left")
        self.angle += 90
        self.check_angle()

    def right(self):
        print("Turn right")
        self.angle -= 90
        self.check_angle()

    def around(self):
        print("Turn around")
        self.angle += 180
        self.check_angle()

    def print_history(self):
        print()
        print("Visited Crossroads:")
        i = 1
        for node in self.history:
            node, speed = node
            x, y = get_coords_from_id(node)
            node = self.map.get_node_by_coords(x, y)
            print("Node " + str(node) + ", arrived with speed " + str(speed))
            i += 1

    def forward(self, speed):
        if self.angle == 0:
            self.x += 1
        elif self.angle == 180:
            self.x -= 1
        elif self.angle == 90:
            self.y -= 1
        elif self.angle == 270:
            self.y += 1
        else:
            return False
        print("Move to node " + str(self.map.get_node_by_coords(self.x, self.y)) + " with speed " + str(speed))
        self.history.append([get_id_from_coords(self.x, self.y), speed])
        return True

    def turn(self, i):
        angle = i * 90
        if self.angle == angle:
            return
        if self.angle + 180 == angle or angle + 180 == self.angle:
            return self.around()
        while angle > self.angle:
            self.left()
        while angle < self.angle:
            self.right()

    def reach_destination(self, end_x, end_y):
        self.visited_nodes = [[self.x, self.y]]
        while self.x != end_x or self.y != end_y:
            roads = self.map.get_adj_edges_of_node(self.x, self.y)
            perception = {"coordinates": [end_x, end_y], "roads": roads}
            res = self.use_knowledge(perception)
            if res == -1:
                return print("Destination crossroad not found.")
        if self.x == end_x or self.y == end_y:
            return print("Agent has reached destination.")

    def use_knowledge(self, perception):
        info = perception["roads"]
        roads = {}
        speeds = {}
        for key in info:
            roads[key] = info[key]['roads']
            image = numpy.array([info[key]['speed'], ])
            predicted_speed = model.predict(image)
            speeds[str(key) + "_" + str(get_id_from_coords(self.x, self.y))] = argmax(predicted_speed) * 10
            speeds[str(get_id_from_coords(self.x, self.y)) + "_" + str(key)] = argmax(predicted_speed) * 10

        data = {'roads': [], 'explored': [get_id_from_coords(self.x, self.y)], 'speeds': speeds}
        for key in roads:
            key_x, key_y = get_coords_from_id(key)
            data['roads'].append([[self.x, self.y], [key_x, key_y]])
            data['explored'].append(key)
            connections = roads[key]
            for i in range(len(connections)):
                coords = connections[i]
                data['roads'].append([[key_x, key_y], coords])
        self.knowledge.tell(data)
        query = {'current': [self.x, self.y], 'destination': perception["coordinates"]}
        action, speed = self.knowledge.ask(query)
        if action == -1:
            return action
        self.turn(action)
        self.forward(speed)
        self.visited_nodes.append([self.x, self.y])
        self.map.show_move(self.x, self.y, self.visited_nodes)


crossroads_number = 25
roadmap = Road(crossroads_number)
roads_to_erase = 16
roadmap.erase_edges(roads_to_erase)
roadmap.draw_road()
car = Agent(roadmap, 1, 3)
car.reach_destination(0, 0)
roadmap.draw_road()
