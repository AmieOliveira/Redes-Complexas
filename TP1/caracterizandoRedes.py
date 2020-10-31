"""
    Trabalho 1 - Caracterizando redes
"""

import networkx as nx
import bisect
import numpy as np
import matplotlib.pyplot as plt

def load_graph(path):
    ftype = path.split(".")[-1]
    if ftype == "gml":
        return nx.read_gml(path, label='id')


def degree_analysis(graph):
    """
        Analiza as informações do grafo com relação ao
        grau de seus vértices
    :param graph: Grafo a ser analizado
    :return:
    """
    d_list = []
    soma = 0

    for v in graph.nodes():
        grau = graph.degree[v]
        bisect.insort(d_list, grau)
        soma += grau

    n = graph.number_of_nodes()

    gmin = d_list[0]
    gmax = d_list[-1]
    media = soma/n
    mediana = d_list[int(n/2)]

    distribuicao = np.array([np.array(range(gmin,gmax+1)), np.zeros(gmax-gmin+1)])
    desvio = 0
    for i in range(n):
        desvio += (d_list[i] - media)**2
        distribuicao[1,d_list[i]-gmin] += 1
    desvio = (desvio/n)**0.5

    i = 0
    while i < distribuicao.shape[1]:
        if distribuicao[1,i] < 0.00000001:
            distribuicao = np.delete(distribuicao, i, 1)
            continue
        i += 1

    return gmin, gmax, media, desvio, mediana, distribuicao


def distance_analysis(graph):
    """
        Analiza as informações do grafo com relação às
        distãncias entre seus vértices
    :param graph: Grafo a ser analizado
    :return:
    """
    # NOTE: Por enquanto só estou considerando arestas com peso inteiro...
    #   Senão vai dar problema
    d_list = []
    soma = 0
    n = graph.number_of_nodes()
    c = int(n*(n-1)/2)

    l_nodes = list(graph.nodes)

    for i in range(n):
        first = i+1
        n1 = l_nodes[i]
        if isinstance(graph, nx.DiGraph) or isinstance(graph, nx.MultiDiGraph):
            first = 0
        for j in range(first, n):
            if i == j:
                continue
            n2 = l_nodes[j]
            try:
                dist = nx.dijkstra_path_length(graph, n1, n2, "distance")
            except nx.NetworkXNoPath:
                c -= 1
                continue
            bisect.insort(d_list, dist)
            soma += dist

    dmin = d_list[0]
    dmax = d_list[-1]
    media = soma/c
    mediana = d_list[int(c/2)]

    distribuicao = np.array([np.array(range(dmin,dmax+1)), np.zeros(dmax-dmin+1)])
    desvio = 0
    for i in range(c):
        desvio += (d_list[i] - media)**2
        distribuicao[1,d_list[i]-dmin] += 1
    desvio = (desvio/c)**0.5

    return dmin, dmax, media, desvio, mediana, distribuicao


def connexity_analysis(graph):
    # NOTE: Só funciona para grafos não direcionados!
    comp = [len(c) for c in sorted(nx.connected_components(graph), key=len)]
    n = len(comp)

    cmin = comp[0]
    cmax = comp[-1]
    media = sum(comp)/n
    mediana = comp[int(n/2)]

    distribuicao = np.array([np.array(range(cmin, cmax + 1)), np.zeros(cmax-cmin+1)])
    desvio = 0
    for i in range(n):
        desvio += (comp[i] - media)**2
        distribuicao[1, comp[i]-cmin] += 1
    desvio = (desvio/n)**0.5

    return cmin, cmax, media, desvio, mediana, distribuicao


def betweenness_analysis(graph):
    n = graph.number_of_nodes()
    cent = sorted(nx.betweenness_centrality(graph).values())

    bmin = cent[0]
    bmax = cent[-1]
    media = sum(cent)/n
    mediana = cent[int(n/2)]

    distrCCDF = [[],[]]
    desvio = 0
    for i in range(n):
        desvio += (cent[i] - media) ** 2

        if cent[i] in distrCCDF[0]:
            continue
        distrCCDF[0] += [cent[i]]
        distrCCDF[1] += [(n-i)/n]
    desvio = (desvio / n) ** 0.5

    return bmin, bmax, media, desvio, mediana, distrCCDF


def closeness_analysis(graph):
    n = graph.number_of_nodes()
    cent = sorted(nx.closeness_centrality(graph).values())

    bmin = cent[0]
    bmax = cent[-1]
    media = sum(cent)/n
    mediana = cent[int(n/2)]

    distrCCDF = [[],[]]
    desvio = 0
    for i in range(n):
        desvio += (cent[i] - media) ** 2

        if cent[i] in distrCCDF[0]:
            continue
        distrCCDF[0] += [cent[i]]
        distrCCDF[1] += [(n-i)/n]
    desvio = (desvio / n) ** 0.5

    return bmin, bmax, media, desvio, mediana, distrCCDF


def clustering_analysis(graph):
    n = graph.number_of_nodes()
    clus = sorted(nx.clustering(graph).values())

    #tot = 3*nx.triangles(graph)/()

    cmin = clus[0]
    cmax = clus[-1]
    media = sum(clus) / n
    mediana = clus[int(n / 2)]

    distrCCDF = [[], []]
    desvio = 0
    for i in range(n):
        desvio += (clus[i] - media) ** 2

        if clus[i] in distrCCDF[0]:
            continue
        distrCCDF[0] += [clus[i]]
        distrCCDF[1] += [(n - i) / n]
    desvio = (desvio / n) ** 0.5

    return cmin, cmax, media, desvio, mediana, distrCCDF






if __name__ == "__main__":
    datapath = "Dados/Newman/karate.gml"


    g = load_graph(datapath)
    #g = nx.path_graph(4)
    #nx.add_path(g, [10, 11, 12])
    #print(g.nodes(), g.edges())

    print(type(g), g.number_of_nodes(), g.number_of_edges())

    print(degree_analysis(g))

    print(distance_analysis(g))

    try:
        print(connexity_analysis(g))
    except nx.NetworkXNotImplemented:
        print("Grafo é direcionado, não dá para fazer esta análise")
        # TODO: Fazer versão fortemente conexa?

    # Analises de contralidade: betweenness e closeness
    print(betweenness_analysis(g))
    print(closeness_analysis(g))

    print(clustering_analysis(g))


    # TODO: Criar gráficos/tabelas comparando diferentes instâncias!!