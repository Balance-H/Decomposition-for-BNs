from pgmpy.models import JunctionTree, DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors import factor_product
from pgmpy.factors.discrete import DiscreteFactor
from Decom_Tree import *
import itertools
import numpy as np
import networkx as nx
from opt_einsum import contract

class Inference_decom:
    """
    Construct a Junction Tree from a pgmpy Bayesian Network
    using a given graph decomposition method, and provide
    einsum-based marginal queries.
    """

    def __init__(self, G, df, R, method):
        """
        G  : pgmpy BayesianNetwork
        df : pandas.DataFrame (用于 MLE)
        R  : list, 查询节点
        """
        self.method = method
        self.data = df
        self.R = R  # 查询节点
        self.model = G
        if self.method in ["decom", "lossless"]:
            self.clique_tree = self._build_clique_tree()
        else:
            raise ValueError(f"未知方法 {self.method}")




    def _build_clique_tree(self):
        if self.method == "decom":
            decom = Graph_Decom(self.model)
            T = decom.PPDD(set(self.R)) # decom tree
        if self.method == "lossless":
            decom = Graph_Decom(nx.moral_graph(self.model))
            T = decom.non_spann_tree() #junction tree

        clique_tree = JunctionTree()

        for c in T.nodes:
            clique_tree.add_node(tuple(sorted(c)))

        for u, v in T.edges():
            clique_tree.add_edge(tuple(sorted(u)), tuple(sorted(v)))

        return clique_tree
    
    def _clique_potentials(self):
        clique_potentials = {}

        for nodes in self.clique_tree.nodes:
            nodes = tuple(sorted(nodes))

            sub_model = DiscreteBayesianNetwork(
                list(self.model.subgraph(list(nodes)).edges)
            )
            sub_model.cpds = []

            sub_model.fit(
                data=self.data[list(nodes)],
                estimator=MaximumLikelihoodEstimator
            )

            factor = None
            for cpd in sub_model.cpds:
                if factor is None:
                    factor = cpd.to_factor()
                else:
                    factor = factor_product(factor, cpd.to_factor())

            clique_potentials[nodes] = factor
            self.clique_tree.add_factors(factor)

        return clique_potentials

    def _sepset_potentials(self):
        sepset_potentials = {}
        sepset_cache = {}

        for (C, D) in self.clique_tree.edges():
            C = tuple(sorted(C))
            D = tuple(sorted(D))

            sepset_nodes = frozenset(set(C) & set(D))

            if sepset_nodes in sepset_cache:
                factor = sepset_cache[sepset_nodes]
            else:
                nodes = list(sepset_nodes)

                sub_model = DiscreteBayesianNetwork(
                    list(self.model.subgraph(nodes).edges)
                )
                sub_model.add_nodes_from(nodes)
                sub_model.cpds = []

                sub_model.fit(
                    data=self.data[nodes],
                    estimator=MaximumLikelihoodEstimator
                )

                factor = None
                for cpd in sub_model.cpds:
                    if factor is None:
                        factor = cpd.to_factor()
                    else:
                        factor = factor_product(factor, cpd.to_factor())

                sepset_cache[sepset_nodes] = factor

            sepset_potentials[frozenset({C, D})] = factor

        return sepset_potentials
    

    def _lossless_potentials(self):
        """
        初始化 clique_potentials 和 sepset_potentials
        使用 lossless 方法（严格 BF 分配逻辑 + 分离子因子分解）
        """
        # 1. 初始化空势函数
        self.clique_potentials = {}
        learn_bn = DiscreteBayesianNetwork()
        learn_bn.add_nodes_from(list(self.model.nodes))
        learn_bn.add_edges_from(list(self.model.edges))
        learn_bn.cpds = []
        learn_bn.fit(self.data, estimator=MaximumLikelihoodEstimator)

        for node in self.clique_tree.nodes():
            vars_in_clique = list(node)
            cards = [learn_bn.get_cpds(v).cardinality[0] for v in vars_in_clique]
            self.clique_potentials[node] = DiscreteFactor(vars_in_clique, cards, np.ones(np.prod(cards)))

        # 2. BF 分配逻辑
        var_to_origin_clique = {}
        for cpd in learn_bn.get_cpds():
            var = cpd.variable
            all_req_vars = {cpd.variable} | set(learn_bn.get_parents(cpd.variable))
            assigned = False
            for node in self.clique_tree.nodes():
                if all_req_vars.issubset(set(node)):
                    var_to_origin_clique[var] = node
                    self.clique_potentials[node] *= cpd.to_factor()
                    assigned = True
                    break
            if not assigned:
                raise ValueError(f"变量 {var} 的 CPD 无法分配！团树中没有任何团同时包含 {all_req_vars}")

        # 3. 分离子方向和 sepset_potentials
        edge_arrows = {}  # {(u,v): {var: target_clique}}
        for u, v in self.clique_tree.edges():
            sep_vars = set(u) & set(v)
            edge_arrows[(u, v)] = {}
            for var in sep_vars:
                origin_c = var_to_origin_clique[var]
                path = nx.shortest_path(self.clique_tree, source=u, target=origin_c)
                edge_arrows[(u, v)][var] = u if v in path else v

        # 4. 推理引擎
        from pgmpy.inference import VariableElimination
        infer = VariableElimination(learn_bn)
        topo_order = list(nx.topological_sort(learn_bn))

        self.sepset_potentials = {}
        for (u, v), arrows in edge_arrows.items():
            s_vars = sorted(list(arrows.keys()), key=lambda x: topo_order.index(x))
            factors = infer.query(variables=s_vars, joint=True)
            self.sepset_potentials[frozenset({u, v})] = factors

            for i, var in enumerate(s_vars):
                target_clique = arrows[var]
                evidence_vars = s_vars[:i]
                phi_joint = infer.query(variables=[var] + evidence_vars, joint=True)
                f_to_add = phi_joint if not evidence_vars else phi_joint / infer.query(variables=evidence_vars, joint=True)
                self.clique_potentials[target_clique] *= f_to_add

        # 注入 clique_tree
        self.clique_tree.factors = []
        for node in self.clique_tree.nodes():
            self.clique_tree.add_factors(self.clique_potentials[node])

        if not self.clique_tree.check_model():
            print("Lossless decomposition fails")


    def query(self):
        """
        Compute marginal distribution over self.R using
        clique and sepset potentials and opt_einsum.
        """
        if self.method == "decom":
            # 2. 初始化团势
            self.clique_potentials = self._clique_potentials()

            # 3. 初始化分离集势
            self.sepset_potentials = self._sepset_potentials()

        else:
            # lossless 使用 moral graph
            self._lossless_potentials()

        cliques = self.clique_tree.nodes
        clique_factors = self.clique_potentials
        sepset_factors = self.sepset_potentials
        edges = self.clique_tree.edges
        query_vars = self.R

        # 1. 所有变量映射到索引
        all_vars = sorted(set(itertools.chain(*cliques)))
        var_int = {v: i for i, v in enumerate(all_vars)}

        einsum_expr = []

        # 2. 团势（分子）
        for C in cliques:
            phi = clique_factors[C]
            einsum_expr.append(phi.values)
            einsum_expr.append([var_int[v] for v in phi.variables])

        # 3. 分离集势（分母）
        for (C, D) in edges:
            psi_S = sepset_factors[frozenset({tuple(sorted(C)), tuple(sorted(D))})]
            eps = 1e-12
            psi_values = np.where(psi_S.values < eps, eps, psi_S.values)
            einsum_expr.append(1.0 / psi_values)
            einsum_expr.append([var_int[v] for v in psi_S.variables])

        # 4. 输出索引
        out_inds = [var_int[v] for v in query_vars]

        return contract(*einsum_expr, out_inds, optimize="greedy")
