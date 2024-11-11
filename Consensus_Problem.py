import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from grakel import Graph, GraphKernel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from grakel.kernels import WeisfeilerLehman, VertexHistogram, Propagation
from collections import deque
from abc import abstractmethod, ABC
from scipy.linalg import eig
from sklearn.kernel_ridge import KernelRidge

df = pd.DataFrame(columns=('graphs', 'probab'))


class Graphs():
    def __init__(self):
        self.z = {}

    @abstractmethod
    def probab(self):
        pass

    def labels_update(self):
        if nx.is_directed(self.G) or nx.is_connected(self.G):
            self.edge_labels = {}
            self.nodes_labels = {}
            a = random.random()
            num1 = int(self.G.number_of_nodes() * a)
            num2 = self.G.number_of_nodes() - random.randint(num1, self.G.number_of_nodes())
            temp_deque = deque()
            for a, b in self.G.adjacency():
                temp_deque.appendleft(a)
            random.shuffle(temp_deque)
            for i in range(len(temp_deque)):
                if i < num1:
                    a = temp_deque[i]
                    self.nodes_labels[a] = 1
                elif i >= num1 and i < num1 + num2:
                    a = temp_deque[i]
                    self.nodes_labels[a] = 2
                else:
                    a = temp_deque[i]
                    self.nodes_labels[a] = 3
            if nx.is_bipartite(self.G):
                set1 = set()
                set2 = set()
                for i, j in self.nodes_labels.items():
                    for k in nx.bipartite.sets(self.G)[0]:
                        if i == k:
                            set1.add(j)
                        else:
                            set2.add(j)
                if len(set1) == 1 or len(set2) == 1:
                    return -1

    def data_concat(self):
        if self.G is not None and (nx.is_directed(self.G) or nx.is_connected(self.G)):
            global df
            print("z", self.z)
            print(self.G)
            graph_grakel = Graph(list(self.G.edges()), node_labels=self.nodes_labels)
            temp_lst = pd.DataFrame([{'graphs': graph_grakel, 'probab': self.z.get(1)}])
            if not temp_lst.empty:
                df = pd.concat([df, temp_lst], ignore_index=True)


class UndirectedGraphs(Graphs):

    def __init__(self):
        super().__init__()

    def graph(self):
        self.G = nx.Graph()
        for i in range(random.randint(5, 100)):
            for j in range(1, random.randint(2, 8)):
                x = random.randint(0, i)
                y = random.randint(0, i)
                if x != y:
                    self.G.add_edge(x, y)

    def erdos_renyi(self, a):
        self.G = nx.erdos_renyi_graph(n=random.randint(2, a), p=random.random())

    def barbell(self, a):
        self.G = nx.barbell_graph(m1=random.randint(2, a), m2=random.randint(1, a))

    def dorogovtsev_goltsev_mendes_graph(self, a):
        self.G = nx.dorogovtsev_goltsev_mendes_graph(random.randint(1, a))

    def mycielski_graph(self, a):
        self.G = nx.mycielski_graph(n=random.randint(3, a))

    def windmill_graph(self, a, b):
        self.G = nx.windmill_graph(n=random.randint(2, a), k=random.randint(3, b))

    def ring_of_cliques(self, a, b):
        self.G = nx.ring_of_cliques(num_cliques=random.randint(2, a), clique_size=random.randint(3, b))

    def planted_partition_graph(self, a, b):
        self.G = nx.planted_partition_graph(random.randint(3, a), random.randint(4, b), random.uniform(0.2, 0.9),
                                            random.uniform(0.3, 0.9))

    def connected_caveman_graph(self, a, b):
        self.G = nx.connected_caveman_graph(random.randint(3, a), random.randint(4, b))

    def random_internet_as_graph(self, a):
        self.G = nx.random_internet_as_graph(random.randint(3, a))

    def random_bipartite_graphs(self, a, b):
        self.G = nx.bipartite.random_graph(random.randint(3, a), random.randint(2, b), p=random.uniform(0.1, 0.9))
        if not nx.is_connected(self.G):
            self.random_bipartite_graphs(a, b)

    def complete_bipartite_graphs(self, a, b):
        self.G = nx.complete_bipartite_graph(random.randint(3, a), random.randint(2, b))

    def tutte_graph(self):
        self.G = nx.tutte_graph()

    def truncated_tetrahedron(self):
        self.G = nx.truncated_tetrahedron_graph()

    def sedgewick_maze_graph(self):
        self.G = nx.sedgewick_maze_graph()

    def petersen_graph(self):
        self.G = nx.petersen_graph()

    def heawood_graph(self):
        self.G = nx.heawood_graph()

    def hoffman_singleton_graph(self):
        self.G = nx.hoffman_singleton_graph()

    def frucht_graph(self):
        self.G = nx.frucht_graph()

    def dodecahedral_graph(self):
        self.G = nx.dodecahedral_graph()

    def desargues_graph(self):
        self.G = nx.desargues_graph()

    def probab(self):
        if nx.is_connected(self.G):
            if nx.is_bipartite(self.G) and nx.is_connected(self.G):
                '''This code from line number 153 to 158 is partially or completely developed using:
                https://stackoverflow.com/a/27085151'''
                c, d = nx.bipartite.sets(self.G)
                self.pos = {}
                self.pos.update((n, (1, i)) for i, n in enumerate(c))
                self.pos.update((n, (2, i)) for i, n in enumerate(d))
                nx.draw(self.G, with_labels=True, labels=self.nodes_labels, pos=self.pos)
                plt.show()
                for a, b in sorted(self.nodes_labels.items()):
                    if a in nx.bipartite.sets(self.G)[0] and b not in self.z:
                        self.z[b] = [dict(self.G.degree())[a]]
                    elif a in nx.bipartite.sets(self.G)[0] and b in self.z:
                        self.z[b][0] = self.z[b][0] + dict(self.G.degree())[a]
                    elif a in nx.bipartite.sets(self.G)[1] and b not in self.z:
                        self.z[b] = [0]
                        self.z[b].append(dict(self.G.degree())[a])
                    elif a in nx.bipartite.sets(self.G)[1] and b in self.z and len(self.z[b]) == 1:
                        self.z[b].append(dict(self.G.degree())[a])
                    elif a in nx.bipartite.sets(self.G)[1] and b in self.z:
                        self.z[b][1] += dict(self.G.degree())[a]
                for a, b in self.z.items():
                    print("Z:", self.z)
                    if len(self.z[a]) == 2:
                        self.z[a] = (self.z[a][0] * self.z[a][1]) / (self.G.number_of_edges() ** 2)
                    else:
                        self.z[a] = (self.z[a][0] * 0) / (self.G.number_of_edges() ** 2)
            else:
                for i in self.nodes_labels.keys():
                    if nx.is_connected(self.G):
                        if self.nodes_labels[i] not in self.z:
                            self.z[self.nodes_labels[i]] = (dict(self.G.degree())[i] / (2 * self.G.number_of_edges()))
                        else:
                            self.z[self.nodes_labels[i]] = self.z[self.nodes_labels[i]] + (
                                        dict(self.G.degree())[i] / (2 * self.G.number_of_edges()))


class Directed_Graphs(Graphs):

    def __init__(self):
        super().__init__()

    def erdos_renyi(self, a):
        self.G = nx.erdos_renyi_graph(n=random.randint(2, a), p=random.random(), directed=True)

    def binomial_graph(self, a):
        self.G = nx.binomial_graph(n=random.randint(2, a), p=random.random(), directed=True)

    def weakly_binomial_graph(self, a):
        self.G = nx.binomial_graph(n=random.randint(2, a), p=random.uniform(0.1, 0.4), directed=True)

    def gaussian_random_partition_graph(self, a):
        self.G = nx.gaussian_random_partition_graph(a, s=random.randint(0, a - 1), v=random.uniform(0.1, 0.9),
                                                    p_in=random.random(), p_out=random.random(), directed=True)
        if not nx.is_connected(self.G.to_undirected()):
            return self.gaussian_random_partition_graph(a)

    def planted_partition_graph(self, a, b):
        self.G = nx.planted_partition_graph(random.randint(3, a), random.randint(4, b), random.uniform(0.2, 0.9),
                                            random.uniform(0.3, 0.9), directed=True)

    def random_bipartite_graphs(self, a, b):
        self.G = nx.bipartite.random_graph(random.randint(3, a), random.randint(2, b), p=random.random(), directed=True)
        if not nx.is_weakly_connected(self.G):
            return self.random_bipartite_graphs(a, b)

    def probab(self):
        g_subgraph = None
        if nx.is_strongly_connected(self.G):
            g_subgraph = self.G
            node_labels_subgraph = self.nodes_labels
        elif not nx.is_strongly_connected(self.G):
            G_condensed = nx.condensation(self.G)
            node_labels_subgraph = dict()
            count = sum(1 for j in dict(G_condensed.out_degree).values() if j == 0)
            for i, j in dict(G_condensed.out_degree).items():
                if count == 1 and j == 0:
                    g_subgraph = self.G.subgraph(list(dict(G_condensed.nodes.data())[i]['members']))
                    for k in list(dict(G_condensed.nodes.data())[i]['members']):
                        node_labels_subgraph[k] = self.nodes_labels[k]
        if g_subgraph is not None and g_subgraph.number_of_edges() >= 1:
            g_subgraph = nx.stochastic_graph(g_subgraph)
            a_m = nx.to_pandas_adjacency(g_subgraph)
            a_m = a_m.reindex(sorted(a_m.columns), axis=1)
            a_m = a_m.reindex(sorted(a_m.columns), axis=0)

            '''This code on line number 238 is partially or completely developed using:
            https://stackoverflow.com/a/33408894'''
            eigenvalues, left_eigenvectors = eig(a_m, right=False, left=True)

            largest_eigenvalue_index = np.argmax(np.isclose(eigenvalues, 1))

            stationary_vector = np.real(left_eigenvectors[:, largest_eigenvalue_index])

            if nx.is_bipartite(g_subgraph):
                '''This code from line number 247 to 252 is partially or completely developed using:
                https://stackoverflow.com/a/27085151'''
                c, d = nx.bipartite.sets(g_subgraph)
                self.pos = {}
                self.pos.update((n, (1, i)) for i, n in enumerate(c))
                self.pos.update((n, (2, i)) for i, n in enumerate(d))
                nx.draw(g_subgraph, with_labels=True, labels=node_labels_subgraph, pos=self.pos)
                plt.show()

                stationary_vector = 2 * stationary_vector / stationary_vector.sum()

                probab = dict()

                for i, j in zip(sorted(node_labels_subgraph), range(len(stationary_vector))):
                    probab[(i, str(node_labels_subgraph[i]))] = stationary_vector[j]
                probab_new = dict()
                temp_set = set()
                for j in nx.bipartite.sets(g_subgraph)[0]:
                    temp_set.add(node_labels_subgraph[j])
                    for k in sorted(probab):
                        if j == k[0] and '0_' + k[1] not in probab_new:
                            probab_new['0_' + k[1]] = probab[k]
                        elif j == k[0] and '0_' + k[1] in probab_new:
                            probab_new['0_' + k[1]] = probab_new['0_' + k[1]] + probab[k]
                if len(temp_set) == 1:
                    return -1

                for a, b in sorted(probab.items()):
                    if a[0] in nx.bipartite.sets(g_subgraph)[0] and a[1] not in self.z:
                        self.z[a[1]] = [b]
                    elif a[0] in nx.bipartite.sets(g_subgraph)[0] and a[1] in self.z:
                        self.z[a[1]][0] = self.z[a[1]][0] + b
                    elif a[0] in nx.bipartite.sets(g_subgraph)[1] and a[1] not in self.z:
                        self.z[a[1]] = [0]
                        self.z[a[1]].append(b)
                    elif a[0] in nx.bipartite.sets(g_subgraph)[1] and a[1] in self.z and len(self.z[a[1]]) == 1:
                        self.z[a[1]].append(b)
                    elif a[0] in nx.bipartite.sets(g_subgraph)[1] and a[1] in self.z:
                        self.z[a[1]][1] += b

                for a, b in self.z.items():
                    if len(self.z[a]) == 2:
                        self.z[a] = (self.z[a][0] * self.z[a][1])
                    else:
                        self.z[a] = (self.z[a][0] * 0)
                self.z = {int(k): v for k, v in self.z.items()}


            else:
                stationary_vector = stationary_vector / stationary_vector.sum()

                probab = dict()

                for i, j in zip(sorted(node_labels_subgraph), range(len(stationary_vector))):
                    probab[(i, str(node_labels_subgraph[i]))] = stationary_vector[j]

                stationary_vector = stationary_vector / stationary_vector.sum()
                probab_new = dict()
                for i in probab:
                    if "1" in i[1] and 1 not in self.z:
                        self.z[1] = probab[i]
                    elif "1" in i[1] and 1 in self.z:
                        self.z[1] = self.z[1] + probab[i]
                    elif "2" in i[1] and 2 not in self.z:
                        self.z[2] = probab[i]
                    elif "2" in i[1] and 2 in self.z:
                        self.z[2] = self.z[2] + probab[i]
                    elif "3" in i[1] and 3 not in self.z:
                        self.z[3] = probab[i]
                    elif "3" in i[1] and 3 in self.z:
                        self.z[3] = self.z[3] + probab[i]
        else:
            return -1


objs = [UndirectedGraphs() for i in range(800)]
a = 19

for obj in range(len(objs)):
    print(obj)
    if obj % a == 0:
        objs[obj].graph()
    elif obj % a == 1:
        objs[obj].erdos_renyi(200)
    elif obj % a == 2:
        objs[obj].barbell(500)
    elif obj % a == 3:
        objs[obj].dorogovtsev_goltsev_mendes_graph(7)
    elif obj % a == 4:
        objs[obj].mycielski_graph(11)
    elif obj % a == 5:
        objs[obj].windmill_graph(75, 65)
    elif obj % a == 6:
        objs[obj].ring_of_cliques(65, 55)
    elif obj % a == 7:
        objs[obj].planted_partition_graph(65, 55)
    elif obj % a == 8:
        objs[obj].connected_caveman_graph(70, 55)
    elif obj % a == 9:
        objs[obj].random_internet_as_graph(45)
    elif obj % a == 10:
        objs[obj].tutte_graph()
    elif obj % a == 11:
        objs[obj].truncated_tetrahedron()
    elif obj % a == 12:
        objs[obj].sedgewick_maze_graph()
    elif obj % a == 13:
        objs[obj].petersen_graph()
    elif obj % a == 14:
        objs[obj].heawood_graph()
    elif obj % a == 15:
        objs[obj].hoffman_singleton_graph()
    elif obj % a == 16:
        objs[obj].frucht_graph()
    elif obj % a == 17:
        objs[obj].dodecahedral_graph()
    elif obj % a == 18:
        objs[obj].desargues_graph()
    t = objs[obj].labels_update()
    if t != -1:
        objs[obj].probab()
        objs[obj].data_concat()
    print()

objs = [UndirectedGraphs() for i in range(800)]
a = 2

for obj in range(len(objs)):
    print(obj)
    if obj % a == 0:
        objs[obj].random_bipartite_graphs(60, 55)
    elif obj % a == 1:
        objs[obj].complete_bipartite_graphs(60, 55)
    t = objs[obj].labels_update()
    if t != -1:
        objs[obj].probab()
        objs[obj].data_concat()
    print()

objs = [Directed_Graphs() for i in range(800)]
a = 6
for obj in range(len(objs)):
    print(obj)
    if obj % a == 0:
        objs[obj].erdos_renyi(200)
    elif obj % a == 1:
        objs[obj].binomial_graph(200)
    elif obj % a == 2:
        objs[obj].weakly_binomial_graph(200)
    elif obj % a == 3:
        objs[obj].gaussian_random_partition_graph(250)
    elif obj % a == 4:
        objs[obj].planted_partition_graph(30, 20)
    elif obj % a == 5:
        objs[obj].random_bipartite_graphs(60, 55)
    if nx.is_connected(objs[obj].G.to_undirected()):
        t = objs[obj].labels_update()
        if t != -1:
            t = objs[obj].probab()
            if t != -1:
                objs[obj].data_concat()
    print()

df[['probab']]

df_copy = df

from sklearn.utils import shuffle

df = shuffle(df)

df['probab'].replace(np.nan, 0, inplace=True)


def data_tts():
    grakel_graphs_new = df['graphs'].to_list()
    y = df['probab']
    x_train, x_test, y_train, y_test = train_test_split(grakel_graphs_new, y, test_size=0.3)
    return x_train, x_test, y_train, y_test


def vertex_histogram(x_train, x_test, y_train, y_test):
    gk = VertexHistogram(normalize=True)
    K_train = gk.fit_transform(x_train)
    clf = SVR(kernel='precomputed')
    clf.fit(K_train, y_train)
    K_test = gk.transform(x_test)
    y_pred = clf.predict(K_test)
    y_pred_train = clf.predict(K_train)
    print("r2 Training/Bias:", r2_score(y_train, y_pred_train), "r2 Testing/Variance:", r2_score(y_test, y_pred))
    print("MSE Training/Bias:", mean_squared_error(y_train, y_pred_train), "MSE Testing/Variance:",
          mean_squared_error(y_test, y_pred))
    print()
    return


x_train, x_test, y_train, y_test = data_tts()
vertex_histogram(x_train, x_test, y_train, y_test)


def vertex_histogram_KRR(x_train, x_test, y_train, y_test):
    gk = VertexHistogram(normalize=True)
    K_train = gk.fit_transform(x_train)
    clf = KernelRidge(kernel='precomputed')
    clf.fit(K_train, y_train)
    K_test = gk.transform(x_test)
    y_pred = clf.predict(K_test)
    y_pred_train = clf.predict(K_train)
    print("r2 Training/Bias:", r2_score(y_train, y_pred_train), "r2 Testing/Variance:", r2_score(y_test, y_pred))
    print("MSE Training/Bias:", mean_squared_error(y_train, y_pred_train), "MSE Testing/Variance:",
          mean_squared_error(y_test, y_pred))
    print()


x_train, x_test, y_train, y_test = data_tts()
vertex_histogram_KRR(x_train, x_test, y_train, y_test)


def WL_Kernel_SVR(x_train, x_test, y_train, y_test):
    r2_scores_WL_test = []
    r2_scores_WL_train = []
    mse_scores_WL_train = []
    mse_scores_WL_test = []
    for i in range(1, 12):
        gk = WeisfeilerLehman(normalize=True, n_iter=i, base_graph_kernel=VertexHistogram, n_jobs=-1)
        K_train = gk.fit_transform(x_train)
        clf = SVR(kernel='precomputed')
        clf.fit(K_train, y_train)
        K_test = gk.transform(x_test)
        y_pred = clf.predict(K_test)
        y_pred_train = clf.predict(K_train)
        print("Training/Bias:", r2_score(y_train, y_pred_train), "Testing/Variance:", r2_score(y_test, y_pred))
        print("MSE Training/Bias:", mean_squared_error(y_train, y_pred_train), "MSE Testing/Variance:",
              mean_squared_error(y_test, y_pred))
        print()
        r2_scores_WL_train.append(r2_score(y_train, y_pred_train))
        r2_scores_WL_test.append(r2_score(y_test, y_pred))
        mse_scores_WL_train.append(mean_squared_error(y_train, y_pred_train))
        mse_scores_WL_test.append(mean_squared_error(y_test, y_pred))
    x = list(range(1, 12))
    plt.plot(x, r2_scores_WL_train, label='Train', color='blue', marker='o', linestyle='-')
    plt.plot(x, r2_scores_WL_test, label='Test', color='green', marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.show()
    plt.plot(x, mse_scores_WL_train, label='Train', color='blue', marker='o', linestyle='-')
    plt.plot(x, mse_scores_WL_test, label='Test', color='green', marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Score')
    plt.legend()
    plt.show()


x_train, x_test, y_train, y_test = data_tts()
WL_Kernel_SVR(x_train, x_test, y_train, y_test)


def WL_Kernel_KR(x_train, x_test, y_train, y_test):
    r2_scores_WL_test = []
    r2_scores_WL_train = []
    mse_scores_WL_train = []
    mse_scores_WL_test = []
    for i in range(1, 12):
        gk = WeisfeilerLehman(n_iter=i, base_graph_kernel=VertexHistogram, normalize=True, n_jobs=-1)
        K_train = gk.fit_transform(x_train)
        clf = KernelRidge(kernel='precomputed')
        clf.fit(K_train, y_train)
        K_test = gk.transform(x_test)
        y_pred = clf.predict(K_test)
        y_pred_train = clf.predict(K_train)
        print("Training/Bias:", r2_score(y_train, y_pred_train), "Testing/Variance:", r2_score(y_test, y_pred))
        print("MSE Training/Bias:", mean_squared_error(y_train, y_pred_train), "MSE Testing/Variance:",
              mean_squared_error(y_test, y_pred))
        print()
        r2_scores_WL_train.append(r2_score(y_train, y_pred_train))
        r2_scores_WL_test.append(r2_score(y_test, y_pred))
        mse_scores_WL_train.append(mean_squared_error(y_train, y_pred_train))
        mse_scores_WL_test.append(mean_squared_error(y_test, y_pred))
    x = list(range(1, 12))
    plt.plot(x, r2_scores_WL_train, label='Train', color='blue', marker='o', linestyle='-')
    plt.plot(x, r2_scores_WL_test, label='Test', color='green', marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.show()
    plt.plot(x, mse_scores_WL_train, label='Train', color='blue', marker='o', linestyle='-')
    plt.plot(x, mse_scores_WL_test, label='Test', color='green', marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Score')
    plt.legend()
    plt.show()


x_train, x_test, y_train, y_test = data_tts()
WL_Kernel_KR(x_train, x_test, y_train, y_test)


def Propagation_SVR(x_train, x_test, y_train, y_test):
    r2_scores_WL_test = []
    r2_scores_WL_train = []
    mse_scores_WL_train = []
    mse_scores_WL_test = []
    for i in range(1, 12):
        gk = Propagation(normalize=True, t_max=i, n_jobs=-1)
        K_train = gk.fit_transform(x_train)
        clf = SVR(kernel='precomputed')
        clf.fit(K_train, y_train)
        K_test = gk.transform(x_test)
        y_pred = clf.predict(K_test)
        y_pred_train = clf.predict(K_train)
        print("Training/Bias:", r2_score(y_train, y_pred_train), "Testing/Variance:", r2_score(y_test, y_pred))
        print("MSE Training/Bias:", mean_squared_error(y_train, y_pred_train), "MSE Testing/Variance:",
              mean_squared_error(y_test, y_pred))
        print()
        r2_scores_WL_train.append(r2_score(y_train, y_pred_train))
        r2_scores_WL_test.append(r2_score(y_test, y_pred))
        mse_scores_WL_train.append(mean_squared_error(y_train, y_pred_train))
        mse_scores_WL_test.append(mean_squared_error(y_test, y_pred))
    x = list(range(1, 12))
    plt.plot(x, r2_scores_WL_train, label='Train', color='blue', marker='o', linestyle='-')
    plt.plot(x, r2_scores_WL_test, label='Test', color='green', marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.show()
    plt.plot(x, mse_scores_WL_train, label='Train', color='blue', marker='o', linestyle='-')
    plt.plot(x, mse_scores_WL_test, label='Test', color='green', marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Score')
    plt.legend()
    plt.show()


x_train, x_test, y_train, y_test = data_tts()
Propagation_SVR(x_train, x_test, y_train, y_test)


def Propagation_KR(x_train, x_test, y_train, y_test):
    r2_scores_WL_test = []
    r2_scores_WL_train = []
    mse_scores_WL_train = []
    mse_scores_WL_test = []
    for i in range(1, 12):
        gk = Propagation(normalize=True, t_max=i, n_jobs=-1)
        K_train = gk.fit_transform(x_train)
        clf = KernelRidge(kernel='precomputed')
        clf.fit(K_train, y_train)
        K_test = gk.transform(x_test)
        y_pred = clf.predict(K_test)
        y_pred_train = clf.predict(K_train)
        print("Training/Bias:", r2_score(y_train, y_pred_train), "Testing/Variance:", r2_score(y_test, y_pred))
        print("MSE Training/Bias:", mean_squared_error(y_train, y_pred_train), "MSE Testing/Variance:",
              mean_squared_error(y_test, y_pred))
        print()
        r2_scores_WL_train.append(r2_score(y_train, y_pred_train))
        r2_scores_WL_test.append(r2_score(y_test, y_pred))
        mse_scores_WL_train.append(mean_squared_error(y_train, y_pred_train))
        mse_scores_WL_test.append(mean_squared_error(y_test, y_pred))
    x = list(range(1, 12))
    plt.plot(x, r2_scores_WL_train, label='Train', color='blue', marker='o', linestyle='-')
    plt.plot(x, r2_scores_WL_test, label='Test', color='green', marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.show()
    plt.plot(x, mse_scores_WL_train, label='Train', color='blue', marker='o', linestyle='-')
    plt.plot(x, mse_scores_WL_test, label='Test', color='green', marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('MSe Score')
    plt.legend()
    plt.show()


x_train, x_test, y_train, y_test = data_tts()
Propagation_KR(x_train, x_test, y_train, y_test)
