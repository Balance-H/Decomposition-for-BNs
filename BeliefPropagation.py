import copy

import networkx as nx
from pgmpy.inference import Inference

from pgmpy.models import (
    DiscreteBayesianNetwork,
    JunctionTree,
)

from pgmpy.inference import VariableElimination

import networkx as nx

from Decom_Tree import *
from DiscreteMarkovNetwork import *


class BeliefPropagation(Inference):
    """
    Class for performing inference using Belief Propagation method.

    Creates a Junction Tree or Clique Tree (JunctionTree class) for the input
    probabilistic graphical model and performs calibration of the junction tree
    so formed using belief propagation.

    Parameters
    ----------
    model: DiscreteBayesianNetwork, DiscreteMarkovNetwork, FactorGraph, JunctionTree
        model for which inference is to performed
    """

    def __init__(self, model):
        super(BeliefPropagation, self).__init__(model)

        if not isinstance(model, JunctionTree):
            self.model = model
            #self.junction_tree = DiscreteMarkovNetwork(self.model.to_markov_model()).to_junction_tree()


            mm_old = model.to_markov_model()
            mm_new = DiscreteMarkovNetwork()
            mm_new.add_nodes_from(mm_old.nodes())
            mm_new.add_edges_from(mm_old.edges())
            mm_new.add_factors(*mm_old.factors)  # <-- 必须手动添加
            self.junction_tree = mm_new.to_junction_tree()

        else:
            self.model = model
            self.junction_tree = copy.deepcopy(model)

        self.clique_beliefs = {}
        self.sepset_beliefs = {}


    def get_cliques(self):
        """
        Returns cliques used for belief propagation.
        """
        return self.junction_tree.nodes()




    def get_clique_beliefs(self):
        """
        Returns clique beliefs. Should be called after the clique tree (or
        junction tree) is calibrated.
        """
        return self.clique_beliefs




    def get_sepset_beliefs(self):
        """
        Returns sepset beliefs. Should be called after clique tree (or junction
        tree) is calibrated.
        """
        return self.sepset_beliefs



    def _update_beliefs(self, sending_clique, receiving_clique, operation):
        """
        This is belief-update method.

        Parameters
        ----------
        sending_clique: node (as the operation is on junction tree, node should be a tuple)
            Node sending the message

        receiving_clique: node (as the operation is on junction tree, node should be a tuple)
            Node receiving the message

        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Takes belief of one clique and uses it to update the belief of the
        neighboring ones.
        """
        sepset = frozenset(sending_clique).intersection(frozenset(receiving_clique))
        sepset_key = frozenset((sending_clique, receiving_clique))

        # \sigma_{i \rightarrow j} = \sum_{C_i - S_{i, j}} \beta_i
        # marginalize the clique over the sepset
        sigma = getattr(self.clique_beliefs[sending_clique], operation)(
            list(frozenset(sending_clique) - sepset), inplace=False
        )

        # \beta_j = \beta_j * \frac{\sigma_{i \rightarrow j}}{\mu_{i, j}}
        self.clique_beliefs[receiving_clique] *= (
            sigma / self.sepset_beliefs[sepset_key]
            if self.sepset_beliefs[sepset_key]
            else sigma
        )

        # \mu_{i, j} = \sigma_{i \rightarrow j}
        self.sepset_beliefs[sepset_key] = sigma

    def _is_converged(self, operation):
        r"""
        Checks whether the calibration has converged or not. At convergence
        the sepset belief would be precisely the sepset marginal.

        Parameters
        ----------
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.
            if operation == marginalize, it checks whether the junction tree is calibrated or not
            else if operation == maximize, it checks whether the junction tree is max calibrated or not

        Formally, at convergence or at calibration this condition would be satisfied for

        .. math:: \sum_{C_i - S_{i, j}} \beta_i = \sum_{C_j - S_{i, j}} \beta_j = \mu_{i, j}

        and at max calibration this condition would be satisfied

        .. math:: \max_{C_i - S_{i, j}} \beta_i = \max_{C_j - S_{i, j}} \beta_j = \mu_{i, j}
        """
        # If no clique belief, then the clique tree is not calibrated
        if not self.clique_beliefs:
            return False

        for edge in self.junction_tree.edges():
            sepset = frozenset(edge[0]).intersection(frozenset(edge[1]))
            sepset_key = frozenset(edge)
            if (
                edge[0] not in self.clique_beliefs
                or edge[1] not in self.clique_beliefs
                or sepset_key not in self.sepset_beliefs
            ):
                return False

            marginal_1 = getattr(self.clique_beliefs[edge[0]], operation)(
                list(frozenset(edge[0]) - sepset), inplace=False
            )
            marginal_2 = getattr(self.clique_beliefs[edge[1]], operation)(
                list(frozenset(edge[1]) - sepset), inplace=False
            )
            if (
                marginal_1 != marginal_2
                or marginal_1 != self.sepset_beliefs[sepset_key]
            ):
                return False
        return True

    def _calibrate_junction_tree(self, operation):
        """
        Generalized calibration of junction tree or clique using belief propagation. This method can be used for both
        calibrating as well as max-calibrating.
        Uses Lauritzen-Spiegelhalter algorithm or belief-update message passing.

        Parameters
        ----------
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Reference
        ---------
        Algorithm 10.3 Calibration using belief propagation in clique tree
        Probabilistic Graphical Models: Principles and Techniques
        Daphne Koller and Nir Friedman.
        """
        # Initialize clique beliefs as well as sepset beliefs
        self.clique_beliefs = {
            clique: self.junction_tree.get_factors(clique)
            for clique in self.junction_tree.nodes()
        }
        self.sepset_beliefs = {
            frozenset(edge): None for edge in self.junction_tree.edges()
        }

        for clique in self.junction_tree.nodes():
            if not self._is_converged(operation=operation):
                neighbors = self.junction_tree.neighbors(clique)
                # update root's belief using neighbor clique's beliefs
                # upward pass
                for neighbor_clique in neighbors:
                    self._update_beliefs(neighbor_clique, clique, operation=operation)
                bfs_edges = nx.algorithms.breadth_first_search.bfs_edges(
                    self.junction_tree, clique
                )
                # update the beliefs of all the nodes starting from the root to leaves using root's belief
                # downward pass
                for edge in bfs_edges:
                    self._update_beliefs(edge[0], edge[1], operation=operation)
            else:
                break


    def calibrate(self):
        """
        Calibration using belief propagation in junction tree or clique tree.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference import BeliefPropagation
        >>> G = DiscreteBayesianNetwork(
        ...     [
        ...         ("diff", "grade"),
        ...         ("intel", "grade"),
        ...         ("intel", "SAT"),
        ...         ("grade", "letter"),
        ...     ]
        ... )
        >>> diff_cpd = TabularCPD("diff", 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD("intel", 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD(
        ...     "grade",
        ...     3,
        ...     [
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ...     ],
        ...     evidence=["diff", "intel"],
        ...     evidence_card=[2, 3],
        ... )
        >>> sat_cpd = TabularCPD(
        ...     "SAT",
        ...     2,
        ...     [[0.1, 0.2, 0.7], [0.9, 0.8, 0.3]],
        ...     evidence=["intel"],
        ...     evidence_card=[3],
        ... )
        >>> letter_cpd = TabularCPD(
        ...     "letter",
        ...     2,
        ...     [[0.1, 0.4, 0.8], [0.9, 0.6, 0.2]],
        ...     evidence=["grade"],
        ...     evidence_card=[3],
        ... )
        >>> G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)
        >>> bp = BeliefPropagation(G)
        >>> bp.calibrate()
        """
        self._calibrate_junction_tree(operation="marginalize")




    def max_calibrate(self):
        """
        Max-calibration of the junction tree using belief propagation.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference import BeliefPropagation
        >>> G = DiscreteBayesianNetwork(
        ...     [
        ...         ("diff", "grade"),
        ...         ("intel", "grade"),
        ...         ("intel", "SAT"),
        ...         ("grade", "letter"),
        ...     ]
        ... )
        >>> diff_cpd = TabularCPD("diff", 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD("intel", 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD(
        ...     "grade",
        ...     3,
        ...     [
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ...     ],
        ...     evidence=["diff", "intel"],
        ...     evidence_card=[2, 3],
        ... )
        >>> sat_cpd = TabularCPD(
        ...     "SAT",
        ...     2,
        ...     [[0.1, 0.2, 0.7], [0.9, 0.8, 0.3]],
        ...     evidence=["intel"],
        ...     evidence_card=[3],
        ... )
        >>> letter_cpd = TabularCPD(
        ...     "letter",
        ...     2,
        ...     [[0.1, 0.4, 0.8], [0.9, 0.6, 0.2]],
        ...     evidence=["grade"],
        ...     evidence_card=[3],
        ... )
        >>> G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)
        >>> bp = BeliefPropagation(G)
        >>> bp.max_calibrate()
        """
        self._calibrate_junction_tree(operation="maximize")



    def _query(
        self, variables, operation, evidence=None, joint=True, show_progress=True
    ):
        """
        This is a generalized query method that can be used for both query and map query.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples
        --------
        >>> from pgmpy.inference import BeliefPropagation
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
        ... )
        >>> model.fit(values)
        >>> inference = BeliefPropagation(model)
        >>> phi_query = inference.query(["A", "B"])

        References
        ----------
        Algorithm 10.4 Out-of-clique inference in clique tree
        Probabilistic Graphical Models: Principles and Techniques Daphne Koller and Nir Friedman.
        """

        is_calibrated = self._is_converged(operation=operation)
        # Calibrate the junction tree if not calibrated
        if not is_calibrated:
            self.calibrate()

        if not isinstance(variables, (list, tuple, set)):
            query_variables = [variables]
        else:
            query_variables = list(variables)
        query_variables.extend(evidence.keys() if evidence else [])

        # Find a tree T' such that query_variables are a subset of scope(T')
        nodes_with_query_variables = set()
        for var in query_variables:
            nodes_with_query_variables.update(
                filter(lambda x: var in x, self.junction_tree.nodes())
            )
        subtree_nodes = nodes_with_query_variables

        # Conversion of set to tuple just for indexing
        nodes_with_query_variables = tuple(nodes_with_query_variables)
        # As junction tree is a tree, that means that there would be only path between any two nodes in the tree
        # thus we can just take the path between any two nodes; no matter there order is
        for i in range(len(nodes_with_query_variables) - 1):
            subtree_nodes.update(
                nx.shortest_path(
                    self.junction_tree,
                    nodes_with_query_variables[i],
                    nodes_with_query_variables[i + 1],
                )
            )
        subtree_undirected_graph = self.junction_tree.subgraph(subtree_nodes)
        # Converting subtree into a junction tree
        if len(subtree_nodes) == 1:
            subtree = JunctionTree()
            subtree.add_node(subtree_nodes.pop())
        else:
            subtree = JunctionTree(subtree_undirected_graph.edges())

        # Selecting a node is root node. Root node would be having only one neighbor
        if len(subtree.nodes()) == 1:
            root_node = list(subtree.nodes())[0]
        else:
            root_node = tuple(
                filter(lambda x: len(list(subtree.neighbors(x))) == 1, subtree.nodes())
            )[0]
        clique_potential_list = [self.clique_beliefs[root_node]]

        # For other nodes in the subtree compute the clique potentials as follows
        # As all the nodes are nothing but tuples so simple set(root_node) won't work at it would update the set with
        # all the elements of the tuple; instead use set([root_node]) as it would include only the tuple not the
        # internal elements within it.
        parent_nodes = set([root_node])
        nodes_traversed = set()
        while parent_nodes:
            parent_node = parent_nodes.pop()
            for child_node in set(subtree.neighbors(parent_node)) - nodes_traversed:
                clique_potential_list.append(
                    self.clique_beliefs[child_node]
                    / self.sepset_beliefs[frozenset([parent_node, child_node])]
                )
                parent_nodes.update([child_node])
            nodes_traversed.update([parent_node])

        # Add factors to the corresponding junction tree
        subtree.add_factors(*clique_potential_list)

        # Sum product variable elimination on the subtree
        variable_elimination = VariableElimination(subtree)
        if operation == "marginalize":
            return variable_elimination.query(
                variables=variables,
                evidence=evidence,
                joint=joint,
                show_progress=show_progress,
            )
        elif operation == "maximize":
            return variable_elimination.map_query(
                variables=variables, evidence=evidence, show_progress=show_progress
            )



    def query(
        self,
        variables,
        evidence=None,
        virtual_evidence=None,
        joint=True,
        show_progress=True,
    ):
        """
        Query method using belief propagation.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        joint: boolean
            If True, returns a Joint Distribution over `variables`.
            If False, returns a dict of distributions over each of the `variables`.

        show_progress: boolean
            If True shows a progress bar.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import BeliefPropagation
        >>> bayesian_model = DiscreteBayesianNetwork(
        ...     [("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")]
        ... )
        >>> cpd_a = TabularCPD("A", 2, [[0.2], [0.8]])
        >>> cpd_r = TabularCPD("R", 2, [[0.4], [0.6]])
        >>> cpd_j = TabularCPD(
        ...     "J", 2, [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]], ["R", "A"], [2, 2]
        ... )
        >>> cpd_q = TabularCPD("Q", 2, [[0.9, 0.2], [0.1, 0.8]], ["J"], [2])
        >>> cpd_l = TabularCPD(
        ...     "L",
        ...     2,
        ...     [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
        ...     ["G", "J"],
        ...     [2, 2],
        ... )
        >>> cpd_g = TabularCPD("G", 2, [[0.6], [0.4]])
        >>> bayesian_model.add_cpds(cpd_a, cpd_r, cpd_j, cpd_q, cpd_l, cpd_g)
        >>> belief_propagation = BeliefPropagation(bayesian_model)
        >>> belief_propagation.query(
        ...     variables=["J", "Q"], evidence={"A": 0, "R": 0, "G": 0, "L": 1}
        ... )
        """
        evidence = evidence if evidence is not None else dict()
        orig_model = self.model.copy()

        # Step 1: Parameter Checks
        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables)
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        # Step 2: If virtual_evidence is provided, modify model and evidence.
        if isinstance(self.model, DiscreteBayesianNetwork) and (
            virtual_evidence is not None
        ):
            self._virtual_evidence(virtual_evidence)
            virt_evidence = {"__" + cpd.variables[0]: 0 for cpd in virtual_evidence}
            return self.query(
                variables=variables,
                evidence={**evidence, **virt_evidence},
                virtual_evidence=None,
                joint=joint,
                show_progress=show_progress,
            )

        # Step 3: Do network pruning.
        if isinstance(self.model, DiscreteBayesianNetwork):
            self.model, evidence = self._prune_bayesian_model(variables, evidence)
        self._initialize_structures()

        # Step 4: Run inference.
        result = self._query(
            variables=variables,
            operation="marginalize",
            evidence=evidence,
            joint=joint,
            show_progress=show_progress,
        )
        self.__init__(orig_model)

        if joint:
            return result.normalize(inplace=False)
        else:
            return result





    def map_query(
        self, variables=None, evidence=None, virtual_evidence=None, show_progress=True
    ):
        """
        MAP Query method using belief propagation. Returns the highest probable
        state in the joint distributon of `variables`.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import BeliefPropagation
        >>> bayesian_model = DiscreteBayesianNetwork(
        ...     [("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")]
        ... )
        >>> cpd_a = TabularCPD("A", 2, [[0.2], [0.8]])
        >>> cpd_r = TabularCPD("R", 2, [[0.4], [0.6]])
        >>> cpd_j = TabularCPD(
        ...     "J", 2, [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]], ["R", "A"], [2, 2]
        ... )
        >>> cpd_q = TabularCPD("Q", 2, [[0.9, 0.2], [0.1, 0.8]], ["J"], [2])
        >>> cpd_l = TabularCPD(
        ...     "L",
        ...     2,
        ...     [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
        ...     ["G", "J"],
        ...     [2, 2],
        ... )
        >>> cpd_g = TabularCPD("G", 2, [[0.6], [0.4]])
        >>> bayesian_model.add_cpds(cpd_a, cpd_r, cpd_j, cpd_q, cpd_l, cpd_g)
        >>> belief_propagation = BeliefPropagation(bayesian_model)
        >>> belief_propagation.map_query(
        ...     variables=["J", "Q"], evidence={"A": 0, "R": 0, "G": 0, "L": 1}
        ... )
        """
        variables = [] if variables is None else variables
        evidence = evidence if evidence is not None else dict()
        common_vars = set(evidence if evidence is not None else []).intersection(
            variables
        )

        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        # TODO:Check the note in docstring. Change that behavior to return the joint MAP
        if not variables:
            variables = list(self.model.nodes())

        # Make a copy of the original model and then replace self.model with it later.
        orig_model = self.model.copy()

        if isinstance(self.model, DiscreteBayesianNetwork) and (
            virtual_evidence is not None
        ):
            self._virtual_evidence(virtual_evidence)
            virt_evidence = {"__" + cpd.variables[0]: 0 for cpd in virtual_evidence}
            return self.map_query(
                variables=variables,
                evidence={**evidence, **virt_evidence},
                virtual_evidence=None,
                show_progress=show_progress,
            )

        if isinstance(self.model, DiscreteBayesianNetwork):
            self.model, evidence = self._prune_bayesian_model(variables, evidence)
        self._initialize_structures()

        final_distribution = self._query(
            variables=variables,
            operation="maximize",
            evidence=evidence,
            joint=True,
            show_progress=show_progress,
        )

        self.__init__(orig_model)

        return final_distribution


