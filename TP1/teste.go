package main

import (
	"fmt"
	"math/rand"

	//"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/graph/path"
	"gonum.org/v1/gonum/graph/topo"
	"gonum.org/v1/gonum/graph/network"
)

func main() {
	n := 5
	p := .8
	rand.Seed(int64(0))

	g := simple.NewUndirectedGraph()
	for i := 0; i < n; i++ {
		//g.AddNode(graph.simple.Node(i))
		for j := i+1; j < n; j++ {
			if rand.Float64() < p {
				g.SetEdge(g.NewEdge(simple.Node(i),simple.Node(j)))
			}
		}
	}

	printGraphInfo(g)

	// Grau
	vg := int64(3)
	fmt.Println( "Grau do vértice ", vg, ": ", g.From( vg ).Len() )

	// Distâncias
	paths := path.DijkstraAllPaths(g) // Exemplo com AllPaths na função FloydWarshall 
	// (varios metodos fora da documentaca)
	for i := 0; i < n; i++ {
		for j := i+1; j < n; j++ {
			fmt.Println("Caminho entre ", i, " e ", j, ": ", paths.Weight(int64(i), int64(j)))
		} 
	}

	// Tamanho das componentes conexas
	componentes := topo.ConnectedComponents(g)
	for i := 0; i < len(componentes); i ++ {
		fmt.Println("Componente ", i+1, ": ", len(componentes[i]), " elementos")
	}

	// Cliques maximais
	maxCliques := topo.BronKerbosch(g)
	fmt.Println("Cliques maximais: ", maxCliques)

	// Betweenness
	btw := network.Betweenness(g)
	fmt.Println("Betweenness do vértice ", vg, ": ", btw[vg])
}


func printGraphInfo(g *simple.UndirectedGraph) {
	fmt.Print("Vértices: ")
	vertices := g.Nodes()
	vertices.Reset()
	totV := vertices.Len()
	for i := 0; i < totV; i++ {
		vertices.Next()
		fmt.Print(vertices.Node(), ", ")
	}
	fmt.Println("")

	fmt.Print("Arestas: ")
	arestas := g.Edges()
	arestas.Reset()
	totA := arestas.Len()
	for i := 0; i < totA; i++ {
		arestas.Next()
		fmt.Print(arestas.Edge(), ", ")
	}
	fmt.Println("\n")
}