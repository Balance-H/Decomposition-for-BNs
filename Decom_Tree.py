# This document includes decomposition, spanning tree construction, and pruning. Updated on 2024.5.20
import random
import networkx as nx
from collections import deque

class Graph_Decom:

    def __init__(self, graph):
        self.graph = graph

    def is_d_separator(self, g, x, y, z):
        try:
            x = {x} if x in g else x
            y = {y} if y in g else y
            z = {z} if z in g else z

            intersection = x & y or x & z or y & z
            if intersection:
                raise nx.NetworkXError(
                    f"The sets are not disjoint, with intersection {intersection}"
                )

            set_v = x | y | z
            if set_v - g.nodes:
                raise nx.NodeNotFound(f"The node(s) {set_v - g.nodes} are not found in G")
        except TypeError:
            raise nx.NodeNotFound("One of x, y, or z is not a node or a set of nodes in G")

        if not nx.is_directed_acyclic_graph(g):
            raise nx.NetworkXError("graph should be directed acyclic")

        # contains -> and <-> edges from starting node T
        forward_deque = deque([])
        forward_visited = set()

        # contains <- and - edges from starting node T
        backward_deque = deque(x)
        backward_visited = set()

        ancestors_or_z = set().union(*[nx.ancestors(g, node) for node in x]) | z | x

        while forward_deque or backward_deque:
            if backward_deque:
                node = backward_deque.popleft()
                backward_visited.add(node)
                if node in y:
                    return False
                if node in z:
                    continue

                # add <- edges to backward deque
                backward_deque.extend(g.pred[node].keys() - backward_visited)
                # add -> edges to forward deque
                forward_deque.extend(g.succ[node].keys() - forward_visited)

            if forward_deque:
                node = forward_deque.popleft()
                forward_visited.add(node)
                if node in y:
                    return False

                # Consider if -> node <- is opened due to ancestor of node in z
                if node in ancestors_or_z:
                    # add <- edges to backward deque
                    backward_deque.extend(g.pred[node].keys() - backward_visited)
                if node not in z:
                    # add -> edges to forward deque
                    forward_deque.extend(g.succ[node].keys() - forward_visited)

        return True
    
    def determine_convex(self,g_m,g,r):
        M = set(g.nodes) - set(r)
        Markov_blanket = nx.node_boundary(g_m, M, r)
        for a in Markov_blanket.copy():
            Markov_blanket.remove(a)
            for b in Markov_blanket:
                if (g.has_edge(a, b) | g.has_edge(b, a)) == False:
                    Z = (nx.ancestors(g, a) | nx.ancestors(g, b)) - M
                    Z.discard(a)
                    Z.discard(b)
                    if self.is_d_separator(g, a, b, Z) == False:
                        return False
        return True
    
    def complete_chordal_graph(self,g):
        H = g.copy()
        alpha = {node: 0 for node in H}
        chords = set()
        weight = {node: 0 for node in H.nodes()}
        unnumbered_nodes = list(H.nodes())
        for i in range(len(H.nodes()), 0, -1):
            # get the node in unnumbered_nodes with the maximum weight
            z = max(unnumbered_nodes, key=lambda node: weight[node])
            unnumbered_nodes.remove(z)
            alpha[z] = i
            update_nodes = []
            for y in unnumbered_nodes:
                if g.has_edge(y, z):
                    update_nodes.append(y)
                else:
                    # y_weight will be bigger than node weights between y and z
                    y_weight = weight[y]
                    lower_nodes = [
                        node for node in unnumbered_nodes if weight[node] < y_weight
                    ]
                    if nx.has_path(H.subgraph(lower_nodes + [z, y]), y, z):
                        update_nodes.append(y)
                        chords.add((z, y))
            # during calculation of paths the weights should not be updated
            for node in update_nodes:
                weight[node] += 1
        H.add_edges_from(chords)
        return H, alpha,weight

    def F_t(self,g):
        H, alpha,weight = self.complete_chordal_graph(g)
        f_t = []
        alpha_up = sorted(alpha, key=alpha.__getitem__)
        for i in range(len(list(g.nodes))-1,0,-1):
            if weight[alpha_up[i-1]] <= weight[alpha_up[i]]:
                f_t.append(alpha_up[i-1])
        return H,f_t,alpha_up
    

    def Decom(self):  # directed graph g_d
        gg = nx.moral_graph(self.graph)
        CH,F_a, pai_2 = self.F_t(gg)
        prime_block = []
        block_A = []
        block_B = []
        S_ft = []
        g = gg.copy()
        for i in range(len(F_a)-1,-1,-1):  # decompose for i = T,...,1. Note that f_1 has been removed in the original paper.
            
            S_ft_1 = list(CH.adj[F_a[i]])  
            index_s = pai_2.index(F_a[i])      
            S_ft_2 = set(element for element in pai_2[index_s + 1:] if element in S_ft_1)        
            S_ft.append(S_ft_2)
        S_ft.reverse()

        # decomposition step
        for i in range(len(F_a)-1,-1,-1):
            
            if self.determine_convex(gg,self.graph,S_ft[i]):  # if convex, note that g is undirected
                V_remove_S_ft = [q for q in g.nodes if q not in S_ft[i]]
                M_c=list(nx.connected_components(nx.subgraph(g, V_remove_S_ft)))
                for k in range(0,len(M_c)):
                    if F_a[i] in list(M_c[k]):
                        brock_A = list(M_c[k])
                        brock_A.extend(S_ft[i])
                        block_B = [p for p in g.nodes if p not in list(M_c[k])]
                        prime_block.append(brock_A)
                        g = nx.subgraph(g, block_B)
                        
        prime_block.append(block_B)
    
        for i in prime_block:
            if isinstance (i,list) == False:
                K = []
                K.append(i)
                prime_block[prime_block.index(i)] = K
        return prime_block
    
    def spann_tree(self):
        nodes_list = self.Decom()
        g_T = nx.Graph()
        for node in nodes_list:
            g_T.add_node(tuple(node), label=node)  # use tuple as node label

        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                intersection = set(nodes_list[i]).intersection(nodes_list[j])
                if intersection:  # if intersection is not empty
                    weight = len(intersection)
                    g_T.add_edge(tuple(nodes_list[i]), tuple(nodes_list[j]), weight=weight, label=','.join(intersection))

        return nx.maximum_spanning_tree(g_T)
    
    def PPDD(self, r):  # pruning
        t = self.spann_tree()
        s = 1  # for repetition
        while s:
            s = 0
            leaf_nodes = [node for node, degree in t.degree() if degree < 2]
            for C in leaf_nodes:
                remaining_nodes = set()
                for i in list(t.nodes):
                    if set(i)!=set(C):
                        remaining_nodes|=set(i)
                if (set(C)&r - remaining_nodes) == set():
                    t.remove_node(C)
                    s = 1

        # simple version only removes leaf nodes           
        # internal_nodes = [node for node, degree in t.degree() if degree > 1]
        
        # for C in internal_nodes:
            # L_C = []  # list of intersecting sets S_i
            # remaining_nodes = set()
            # for i in list(t.nodes):
                # if set(i)!=set(C):
                    # remaining_nodes|=set(i)
            # if (set(C)&r - remaining_nodes) == set():
                # for edge, attributes in t.adj[C].items():
                    # label_str = attributes['label']
                    # labels = set(label_str.split(','))
                    # L_C.append(labels)
                # for j in L_C.copy():
                    # L_C.remove(j)
                    # for k in L_C:          
                        # if len(j&k)>0 or (self.is_d_separator(self.graph, j-k, k, set()) == False):
                            # break
                    # else:
                        # continue
                    # break
                # else:  # if pairwise independent
                    # t.remove_node(C)
        # s = 1  # for repetition
        # while s:
            # s = 0
            # leaf_nodes = [node for node, degree in t.degree() if degree < 2]
            # for C in leaf_nodes:
                # remaining_nodes = set()
                # for i in list(t.nodes):
                    # if set(i)!=set(C):
                        # remaining_nodes|=set(i)
                # if (set(C)&r - remaining_nodes) == set():
                    # t.remove_node(C)
                    # s = 1
                                
        return t
