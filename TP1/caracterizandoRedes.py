"""
    Trabalho 1 - Caracterizando redes
"""

import networkx as nx
import bisect
import numpy as np
import matplotlib.pyplot as plt
import json as js
#import pandas as pd

def load_graph(path):
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
    elif ftype == "edges":
        return nx.read_weighted_edgelist(path)
    else:
        print("Loading not implemented for given format")


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
        distribuicao[1,d_list[i]-dmin] += 1/c
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
    outFile = "Dados/estatisticas.txt"
    datapaths = ["Dados/Newman/karate.gml", "Dados/Newman/lesmis.gml",
                 "Dados/Newman/netscience.gml",
                 "Dados/ICON/starwars-full-interactions.json",
                 "Dados/Outros/aves-songbird-social.edges"]
    nomes = ["Karate Club", "Les Miserables", "Ciência das Redes", "Star Wars", "Pássaros Canoros"]
    styles = ['--o', '--s', '--^', '--*', '--X'] # D,p

    f = open(outFile, "w")

    figDeg = plt.figure("Grau")
    plt.title("PDFs empíricas dos graus das redes")
    figDist = plt.figure("Distância")
    plt.title("PDFs empíricas das distâncias nas redes")
    figCon = plt.figure("Componentes Conexas")
    plt.title("PDFs empíricas dos tamanhos das componentes conexas das redes")
    figBet = plt.figure("Betweenness")
    plt.title("CCDFs empíricas de contralidade por betweenness nas redes")
    figClose = plt.figure("Closeness")
    plt.title("CCDFs empíricas de contralidade por closeness nas redes")
    figClus = plt.figure("Clusterização")
    plt.title("CCDFs empíricas da clusterização dos vértices das redes")

    for pathIdx in range(len(datapaths)):
        g = load_graph(datapaths[pathIdx])

        print(nomes[pathIdx], ": (", g.number_of_nodes(), g.number_of_edges(), ")")

        f.write(f"REDE: {nomes[pathIdx]} - {g.number_of_nodes()} vértices X "
                f"{g.number_of_edges()} arestas ({datapaths[pathIdx]})\n")

        deg = degree_analysis(g)
        f.write(f"Grau: "
                f"\tMínimo: {deg[0]}\n"
                f"\tMáximo: {deg[1]}\n"
                f"\tMédia: {deg[2]} +- {deg[3]}\n"
                f"\tMediana: {deg[4]}\n"
                f"{deg[5]} (PDF)\n\n")
        plt.figure(figDeg.number)
        plt.plot(deg[5][0], deg[5][1], styles[pathIdx], linewidth=.8, label=nomes[pathIdx])
        if nomes[pathIdx] == nomes[-1]:
            plt.legend()

        dist = distance_analysis(g)
        f.write(f"Distância: \n"
                f"\tMínima: {dist[0]}\n"
                f"\tMáxima: {dist[1]}\n"
                f"\tMédia: {dist[2]} +- {dist[3]}\n"
                f"\tMediana: {dist[4]}\n"
                f"{dist[5]} (PDF)\n\n")
        plt.figure(figDist.number)
        plt.plot(dist[5][0], dist[5][1], styles[pathIdx], linewidth=.8, label=nomes[pathIdx])
        if nomes[pathIdx] == nomes[-1]:
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
            plt.plot(con[6][0], con[6][1], styles[pathIdx], linewidth=.8, label=nomes[pathIdx])
            plt.legend()
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
        plt.plot(bet[5][0], bet[5][1], styles[pathIdx], linewidth=.8, label=nomes[pathIdx])
        if nomes[pathIdx] == nomes[-1]:
            plt.legend()
        close = closeness_analysis(g)
        f.write(f"Closeness: \t(Contralidade 2)\n"
                f"\tMínimo: {close[0]}\n"
                f"\tMáximo: {close[1]}\n"
                f"\tMédia: {close[2]} +- {close[3]}\n"
                f"\tMediana: {close[4]}\n"
                f"{close[5]} (CCDF)\n\n")
        plt.figure(figClose.number)
        plt.plot(close[5][0], close[5][1], styles[pathIdx], linewidth=.8, label=nomes[pathIdx])
        if nomes[pathIdx] == nomes[-1]:
            plt.legend()

        try:
            clust = clustering_analysis(g)
            f.write(f"Clustering: \n"
                    f"\tMínimo: {bet[0]}\n"
                    f"\tMáximo: {bet[1]}\n"
                    f"\tMédia: {bet[2]} +- {bet[3]}\n"
                    f"\tMediana: {bet[4]}\n"
                    f"{bet[5]} (CCDF)\n")
            plt.figure(figClus.number)
            plt.plot(clust[5][0], clust[5][1], styles[pathIdx], linewidth=.8, label=nomes[pathIdx])
            plt.legend()
        except nx.NetworkXNotImplemented:
            print("Grafo é 'multigraph', análise de clusterização não realizada")

        f.write("--------------------------\n\n")


    plt.figure(figClus.number)
    plt.savefig("Dados/distribuicao-clusterizacao.pdf")
    plt.figure(figClose.number)
    plt.savefig("Dados/distribuicao-closeness.pdf")
    plt.figure(figBet.number)
    plt.savefig("Dados/distribuicao-betweenness.pdf")
    plt.figure(figCon.number)
    plt.savefig("Dados/distribuicao-compConexas.pdf")
    plt.figure(figDist.number)
    plt.savefig("Dados/distribuicao-distancias.pdf")
    plt.figure(figDeg.number)
    plt.savefig("Dados/distribuicao-graus.pdf")
    #plt.show()

    f.close()