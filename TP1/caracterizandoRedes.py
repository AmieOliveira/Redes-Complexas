"""
    Trabalho 1 - Caracterizando redes
"""

import networkx as nx
import bisect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import json as js
#import pandas as pd

def load_graph(path, directed=False, weighted=False):
    ftype = path.split(".")[-1]
    if ftype == "gml":
        return nx.read_gml(path, label='id')
    #elif ftype == "csv":
    #    df = pd.read_csv(path)
    #    print(df.head(10))
    #    Graphtype = nx.Graph()
    #    # TODO: Para CSVs, parece que eu preciso setar manualmente o que são atributos...
    #    return nx.from_pandas_edgelist(df, create_using=Graphtype)
    elif ftype == "json":
        with open(path) as f:
            data = js.load(f)
            return nx.readwrite.json_graph.node_link_graph(data)
    elif ftype in ["edges", "txt"]:
        if directed:
            print("Loading not implemented for directed edgelists")
            return
        else:
            if weighted:
                return nx.read_weighted_edgelist(path)
            else:
                return nx.read_edgelist(path)
    else:
        print("Loading not implemented for given format")
        return


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
        distribuicao[1,d_list[i]-gmin] += 1/n
    desvio = (desvio/n)**0.5

    #i = 0
    #while i < distribuicao.shape[1]:
    #    if distribuicao[1,i] < 0.00000001:
    #        distribuicao = np.delete(distribuicao, i, 1)
    #        continue
    #    i += 1

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
                dist = nx.shortest_path_length(graph, n1, n2)
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
        distribuicao[1,d_list[i]-dmin] += 1/c
    desvio = (desvio/c)**0.5

    return dmin, dmax, media, desvio, mediana, distribuicao, c


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
        distribuicao[1, comp[i]-cmin] += 1/n
    desvio = (desvio/n)**0.5

    return n, cmin, cmax, media, desvio, mediana, distribuicao


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
    cmap = plt.get_cmap("tab10")

    outFile = "Dados/estatisticas.txt"

    datasets = { "Karate Club": {'path': "Dados/Newman/karate.gml",
                                 'weighted': False, 'directed': False,
                                 'style': '--o', 'color':'tab:blue'},
                 "Les Miserables": {'path': "Dados/Newman/lesmis.gml",
                                    'weighted': True, 'directed': False,
                                    'style': '--s', 'color':'tab:orange'},
                 "Ciência das Redes": {'path': "Dados/Newman/netscience.gml",
                                       'weighted': True, 'directed': False,
                                       'style': '--^', 'color':'tab:green'},
                 "Star Wars": {'path': "Dados/ICON/starwars-full-interactions.json",
                               'weighted': True, 'directed': False,
                               'style': '--*', 'color':'tab:red'},
                 #"Pássaros Canoros": {'path': "Dados/Outros/aves-songbird-social.edges",
                 #                     'weighted': True, 'directed': False,
                 #                     'style': '--X', 'color':'tab:purple'},
                 #"Actor": {'path': "Dados/Barabasi/actor.edgelist.txt",
                 #          'weighted': False, 'directed': False,
                 #          'style': '--D', 'color': 'tab:brown'},
                 }  # Styles: D,p - Colors: tab:brown, tab:pink, tab:gray, tab:olive, tab:cyan


    # --------------------------------
    lastKey = list(datasets.keys())[-1]

    f = open(outFile, "w")

    figDeg = plt.figure("Grau", figsize=(10,5))
    plt.title("PDFs empíricas dos graus das redes")
    figDegLog = plt.figure("Grau (Log)", figsize=(10,5))
    plt.title("PDFs empíricas dos graus das redes (Escala Log)")
    axDegLog = figDegLog.add_subplot(111)
    figDist = plt.figure("Distância", figsize=(10,5))
    plt.title("PDFs empíricas das distâncias nas redes")
    figDistLog = plt.figure("Distância (Log)", figsize=(10,5))
    plt.title("PDFs empíricas das distâncias nas redes")
    axDistLog = figDistLog.add_subplot(111)
    figCon = plt.figure("Componentes Conexas", figsize=(10,5))
    plt.title("PDFs empíricas dos tamanhos das componentes conexas das redes")
    figConLog = plt.figure("Componentes Conexas (Log)", figsize=(10, 5))
    plt.title("PDFs empíricas dos tamanhos das componentes conexas das redes")
    axConLog = figConLog.add_subplot(111)
    figBet = plt.figure("Betweenness", figsize=(10,5))
    plt.title("CCDFs empíricas de contralidade por betweenness nas redes")
    figBetLog = plt.figure("Betweenness (Log)", figsize=(10, 5))
    plt.title("CCDFs empíricas de contralidade por betweenness nas redes")
    axBetLog = figBetLog.add_subplot(111)
    figClose = plt.figure("Closeness", figsize=(10,5))
    plt.title("CCDFs empíricas de contralidade por closeness nas redes")
    figCloseLog = plt.figure("Closeness (Log)", figsize=(10, 5))
    plt.title("CCDFs empíricas de contralidade por closeness nas redes")
    axCloseLog = figCloseLog.add_subplot(111)
    figClus = plt.figure("Clusterização", figsize=(10,5))
    plt.title("CCDFs empíricas da clusterização dos vértices das redes")
    figClusLog = plt.figure("Clusterização (Log)", figsize=(10, 5))
    plt.title("CCDFs empíricas da clusterização dos vértices das redes")
    axClusLog = figClusLog.add_subplot(111)

    for key in datasets.keys():
        g = load_graph(datasets[key]['path'],
                       weighted=datasets[key]['weighted'],
                       directed=datasets[key]['directed'])

        print(key, ": (", g.number_of_nodes(), g.number_of_edges(), ")")

        f.write(f"REDE: {key} - {g.number_of_nodes()} vértices X "
                f"{g.number_of_edges()} arestas ({datasets[key]['path']})\n")

        deg = degree_analysis(g)
        f.write(f"Grau: "
                f"\tMínimo: {deg[0]}\n"
                f"\tMáximo: {deg[1]}\n"
                f"\tMédia: {deg[2]} +- {deg[3]}\n"
                f"\tMediana: {deg[4]}\n"
                f"{deg[5]} (PDF)\n\n")
        plt.figure(figDeg.number)
        plt.plot(deg[5][0], deg[5][1], datasets[key]['style'],
                 color= datasets[key]['color'], linewidth=.8, label=key)
        axDegLog.plot(deg[5][0], deg[5][1], datasets[key]['style'],
                      color= datasets[key]['color'], linewidth=0.5, label=key)

        if key == lastKey:
            plt.xlabel("Grau do vértice")
            plt.legend()
            plt.grid(True)
            plt.figure(figDegLog.number)
            plt.xlabel("Grau do vértice")
            axDegLog.grid(True)
            axDegLog.set_yscale("logit")
            axDegLog.set_xscale("log")
            axDegLog.yaxis.set_minor_formatter(NullFormatter())
            plt.legend()

        dist = distance_analysis(g)
        f.write(f"Distância: ({dist[6]} pares alcançáveis)\n"
                f"\tMínima: {dist[0]}\n"
                f"\tMáxima: {dist[1]}\n"
                f"\tMédia: {dist[2]} +- {dist[3]}\n"
                f"\tMediana: {dist[4]}\n"
                f"{dist[5]} (PDF)\n\n")
        plt.figure(figDist.number)
        plt.plot(dist[5][0], dist[5][1], datasets[key]['style'],
                 color= datasets[key]['color'], linewidth=.8, label=key)
        axDistLog.plot(dist[5][0], dist[5][1], datasets[key]['style'],
                        color= datasets[key]['color'], linewidth=.5, label=key)
        if key == lastKey:
            plt.xlabel("Distância entre vértices")
            plt.legend()
            plt.grid(True)
            plt.figure(figDistLog.number)
            plt.xlabel("Distância entre vértices")
            axDistLog.grid(True)
            axDistLog.set_yscale("log")
            axDistLog.set_xscale("log")
            axDistLog.yaxis.set_minor_formatter(NullFormatter())
            plt.legend()

        try:
            con = connexity_analysis(g)
            f.write(f"Componentes conexas: {con[0]} componentes independentes \n"
                    f"\tMínima: {con[1]}\n"
                    f"\tMáxima: {con[2]}\n"
                    f"\tMédia: {con[3]} +- {con[4]}\n"
                    f"\tMediana: {con[5]}\n"
                    f"{con[6]} (PDF)\n\n")
            plt.figure(figCon.number)
            plt.plot(con[6][0], con[6][1], datasets[key]['style'],
                 color= datasets[key]['color'], linewidth=.8, label=key)
            plt.legend()
            plt.xlabel("Tamanho da componente conexa")
            plt.grid(True)

            plt.figure(figConLog.number)
            axConLog.plot(con[6][0], con[6][1], datasets[key]['style'],
                          color= datasets[key]['color'], linewidth=0.5, label=key)
            axConLog.grid(True)
            axConLog.set_xscale("log")
            axConLog.set_yscale("log")
            axConLog.yaxis.set_minor_formatter(NullFormatter())
            axConLog.legend(loc="right")
            plt.xlabel("Tamanho da componente conexa")

        except nx.NetworkXNotImplemented:
            print("Grafo é direcionado, análise de componenetes conexas nao realizada")
            # TODO: Fazer versão fortemente conexa?

        # Analises de contralidade: betweenness e closeness
        bet = betweenness_analysis(g)
        f.write(f"Betweenness: \t(Contralidade 1)\n"
                f"\tMínimo: {bet[0]}\n"
                f"\tMáximo: {bet[1]}\n"
                f"\tMédia: {bet[2]} +- {bet[3]}\n"
                f"\tMediana: {bet[4]}\n"
                f"{bet[5]} (CCDF)\n\n")
        plt.figure(figBet.number)
        plt.plot(bet[5][0], bet[5][1], datasets[key]['style'],
                 color= datasets[key]['color'], linewidth=.8, label=key)
        axBetLog.plot(bet[5][0], bet[5][1], datasets[key]['style'],
                      color= datasets[key]['color'], linewidth=0.5, label=key)
        if key == lastKey:
            plt.xlabel("Betweenness do vértice")
            plt.legend()
            plt.grid(True)
            plt.figure(figBetLog.number)
            plt.xlabel("Betweenness do vértice")
            axBetLog.grid(True)
            axBetLog.set_yscale("log")
            axBetLog.set_xscale("log")
            axBetLog.yaxis.set_minor_formatter(NullFormatter())
            plt.legend()

        close = closeness_analysis(g)
        f.write(f"Closeness: \t(Contralidade 2)\n"
                f"\tMínimo: {close[0]}\n"
                f"\tMáximo: {close[1]}\n"
                f"\tMédia: {close[2]} +- {close[3]}\n"
                f"\tMediana: {close[4]}\n"
                f"{close[5]} (CCDF)\n\n")
        plt.figure(figClose.number)
        plt.plot(close[5][0], close[5][1], datasets[key]['style'],
                 color= datasets[key]['color'], linewidth=.8, label=key)
        axCloseLog.plot(close[5][0], close[5][1], datasets[key]['style'],
                        color= datasets[key]['color'], linewidth=.5, label=key)
        if key == lastKey:
            plt.xlabel("Closeness do vértice")
            plt.legend()
            plt.grid(True)
            plt.figure(figCloseLog.number)
            plt.xlabel("Closeness do vértice")
            axCloseLog.grid(True)
            axCloseLog.set_yscale("log")
            axCloseLog.set_xscale("log")
            axCloseLog.yaxis.set_minor_formatter(NullFormatter())
            plt.legend()

        try:
            clust = clustering_analysis(g)
            f.write(f"Clustering: \n"
                    f"\tMínimo: {clust[0]}\n"
                    f"\tMáximo: {clust[1]}\n"
                    f"\tMédia: {clust[2]} +- {clust[3]}\n"
                    f"\tMediana: {clust[4]}\n"
                    f"{clust[5]} (CCDF)\n")
            plt.figure(figClus.number)
            plt.plot(clust[5][0], clust[5][1], datasets[key]['style'],
                 color= datasets[key]['color'], linewidth=.8, label=key)
            axClusLog.plot(clust[5][0], clust[5][1], datasets[key]['style'],
                           color=datasets[key]['color'], linewidth=.5, label=key)
            plt.xlabel("Clusterização do vértice")
            plt.legend()
            plt.grid(True)
            plt.figure(figClusLog.number)
            plt.xlabel("Clusterização do vértice")
            axClusLog.grid(True)
            axClusLog.set_yscale("log")
            axClusLog.set_xscale("log")
            axClusLog.yaxis.set_minor_formatter(NullFormatter())
            plt.legend()
        except nx.NetworkXNotImplemented:
            print("Grafo é 'multigraph', análise de clusterização não realizada")

        f.write("--------------------------\n\n")


    plt.figure(figClus.number)
    plt.savefig("Dados/distribuicao-clusterizacao.pdf")
    plt.figure(figClusLog.number)
    plt.savefig("Dados/distribuicao-clusterizacao-log.pdf")
    plt.figure(figClose.number)
    plt.savefig("Dados/distribuicao-closeness.pdf")
    plt.figure(figCloseLog.number)
    plt.savefig("Dados/distribuicao-closeness-log.pdf")
    plt.figure(figBet.number)
    plt.savefig("Dados/distribuicao-betweenness.pdf")
    plt.figure(figBetLog.number)
    plt.savefig("Dados/distribuicao-betweenness-log.pdf")
    plt.figure(figConLog.number)
    plt.savefig("Dados/distribuicao-compConexas-log.pdf")
    plt.figure(figCon.number)
    plt.savefig("Dados/distribuicao-compConexas.pdf")
    plt.figure(figDist.number)
    plt.savefig("Dados/distribuicao-distancias.pdf")
    plt.figure(figDistLog.number)
    plt.savefig("Dados/distribuicao-distancias-log.pdf")
    plt.figure(figDeg.number)
    plt.savefig("Dados/distribuicao-graus.pdf")
    plt.figure(figDegLog.number)
    plt.savefig("Dados/distribuicao-graus-log.pdf")
    #plt.show()

    f.close()
