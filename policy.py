from enum import Enum, auto
from pacsltk import perfmodel

from faas import Node

import conf

import hashlib
import bisect


COLD_START_PROB_INITIAL_GUESS = 0.0


class SchedulerDecision(Enum):
    EXEC = 1
    OFFLOAD_CLOUD = 2
    OFFLOAD_EDGE = 3
    DROP = 4


class ColdStartEstimation(Enum):
    NO = auto()
    NAIVE = auto()
    NAIVE_PER_FUNCTION = auto()
    PACS = auto()
    FULL_KNOWLEDGE = auto()

    @classmethod
    def from_string(cls,s):
        s = s.lower()
        if s == "no":
            return ColdStartEstimation.NO
        elif s == "naive" or s == "":
            return ColdStartEstimation.NAIVE
        elif s == "naive-per-function":
            return ColdStartEstimation.NAIVE_PER_FUNCTION
        elif s == "pacs":
            return ColdStartEstimation.PACS
        elif s == "full-knowledge":
            return ColdStartEstimation.FULL_KNOWLEDGE
        return None


class Policy:

    def __init__(self, simulation, node):
        self.simulation = simulation
        self.node = node
        self.__edge_peers = None
        self.budget = simulation.config.getfloat(conf.SEC_POLICY, conf.HOURLY_BUDGET, fallback=-1.0)
        self.local_budget = self.budget
        if simulation.config.getboolean(conf.SEC_POLICY, conf.SPLIT_BUDGET_AMONG_EDGE_NODES, fallback=False):
            nodes = len(simulation.infra.get_edge_nodes())
            self.local_budget = self.budget / nodes


    def schedule(self, function, qos_class, offloaded_from):
        pass

    def update(self):
        pass

    def can_execute_locally(self, f, reclaim_memory=True):
        if f in self.node.warm_pool or self.node.curr_memory >= f.memory:
            return True
        if reclaim_memory:
            reclaimed = self.node.warm_pool.reclaim_memory(f.memory - self.node.curr_memory)
            self.node.curr_memory += reclaimed
        return self.node.curr_memory >= f.memory

    def _get_edge_peers (self):
        if self.__edge_peers is None:
            # TODO: need to refresh over time?
            self.__edge_peers = self.simulation.infra.get_neighbors(self.node, self.simulation.node_choice_rng, self.simulation.max_neighbors)
        return self.__edge_peers

    def _get_edge_peers_probabilities (self):
        peers = self._get_edge_peers()
        for peer in peers:
            if peer.curr_memory < 0.0:
                print(peer)
                print(peer.curr_memory)
            if peer.peer_exposed_memory_fraction < 0.0:
                print(peer)
                print(peer.peer_exposed_memory_fraction)
            assert(peer.curr_memory*peer.peer_exposed_memory_fraction >= 0.0)
        total_memory = sum([x.curr_memory*x.peer_exposed_memory_fraction for x in peers])
        if total_memory > 0.0:
            probs = [x.curr_memory*x.peer_exposed_memory_fraction/total_memory for x in peers]
        else:
            n = len(peers)
            probs = [1.0/n for x in peers]
        return probs, peers

    # Picks a node for Edge offloading
    def pick_edge_node (self, fun, qos):
        # Pick peers based on resource availability
        probs, peers = self._get_edge_peers_probabilities()
        if len(peers) < 1:
            return None
        return self.simulation.node_choice_rng.choice(peers, p=probs)


class BasicPolicy(Policy):

    def schedule(self, f, c, offloaded_from):
        if self.can_execute_locally(f):
            return (SchedulerDecision.EXEC, None)
        else:
            return (SchedulerDecision.OFFLOAD_CLOUD, None)


class BasicBudgetAwarePolicy(Policy):

    def schedule(self, f, c, offloaded_from):
        budget_ok = self.simulation.stats.cost / self.simulation.t * 3600 < self.budget

        if self.can_execute_locally(f):
            return (SchedulerDecision.EXEC, None)
        elif self.simulation.stats.cost / self.simulation.t * 3600 < self.budget:
            return (SchedulerDecision.OFFLOAD_CLOUD, None)
        else:
            return (SchedulerDecision.DROP, None)


class BasicEdgePolicy(Policy):

    def schedule(self, f, c, offloaded_from):
        if self.can_execute_locally(f):
            return (SchedulerDecision.EXEC, None)
        elif len(offloaded_from) == 0:
            return (SchedulerDecision.OFFLOAD_EDGE, self.pick_edge_node(f,c))
        else:
            return (SchedulerDecision.DROP, None)


class CloudPolicy(Policy):

    def schedule(self, f, c, offloaded_from):
        if self.can_execute_locally(f):
            return (SchedulerDecision.EXEC, None)
        else:
            return (SchedulerDecision.DROP, None)


class GreedyPolicy(Policy):

    def __init__(self, simulation, node):
        super().__init__(simulation, node)
        self.cold_start_prob = {}
        self.cold_start_prob_cloud = {}
        self.estimated_service_time = {}
        self.estimated_service_time_cloud = {}

        self.local_cold_start_estimation = ColdStartEstimation.from_string(self.simulation.config.get(conf.SEC_POLICY, conf.LOCAL_COLD_START_EST_STRATEGY, fallback=""))
        self.cloud_cold_start_estimation = ColdStartEstimation.from_string(self.simulation.config.get(conf.SEC_POLICY, conf.CLOUD_COLD_START_EST_STRATEGY, fallback=""))

        # OLD: cloud_region = node.region.default_cloud
        #self.cloud = self.simulation.node_choice_rng.choice(self.simulation.infra.get_region_nodes(cloud_region), 1)[0]

        # Pick the closest cloud node
        nodes_w_lat = [(_n,simulation.infra.get_latency(node,_n)) for _n in simulation.infra.get_cloud_nodes()]
        self.cloud = sorted(nodes_w_lat, key=lambda x: x[1])[0][0]

    def _estimate_latency (self, f, c):
        if self.local_cold_start_estimation == ColdStartEstimation.FULL_KNOWLEDGE:
            if f in self.node.warm_pool:
                self.cold_start_prob[(f, self.node)] = 0
            else:
                self.cold_start_prob[(f, self.node)] = 1
        if self.cloud_cold_start_estimation == ColdStartEstimation.FULL_KNOWLEDGE:
            if f in self.cloud.warm_pool:
                self.cold_start_prob[(f, self.cloud)] = 0
            else:
                self.cold_start_prob[(f, self.cloud)] = 1

        latency_local = self.estimated_service_time.get(f, 0) + \
                        self.cold_start_prob.get((f, self.node), 1) * \
                        self.simulation.init_time[(f,self.node)]

        latency_cloud = self.estimated_service_time_cloud.get(f, 0) +\
                2 * self.simulation.infra.get_latency(self.node, self.cloud) + \
                        self.cold_start_prob.get((f, self.cloud), 1) * self.simulation.init_time[(f,self.cloud)] +\
                        f.inputSizeMean*8/1000/1000/self.simulation.infra.get_bandwidth(self.node, self.cloud)
        return (latency_local, latency_cloud)

    def schedule(self, f, c, offloaded_from):
        latency_local, latency_cloud = self._estimate_latency(f,c)

        if self.can_execute_locally(f) and latency_local < latency_cloud:
            return (SchedulerDecision.EXEC, None)
        else:
            return (SchedulerDecision.OFFLOAD_CLOUD, self.cloud)

    def update_cold_start (self, stats):
        #
        # LOCAL NODE
        #
        if self.local_cold_start_estimation == ColdStartEstimation.PACS:
            for f in self.simulation.functions:
                total_arrival_rate = max(0.001, sum([stats.arrivals.get((f,x,self.node), 0.0) for x in self.simulation.classes])/self.simulation.t)
                props1, _ = perfmodel.get_sls_warm_count_dist(total_arrival_rate,
                                                            self.estimated_service_time[f],
                                                            self.estimated_service_time[f] + self.simulation.init_time[(f,self.node)],
                                                            self.simulation.expiration_timeout)
                self.cold_start_prob[(f, self.node)] = props1["cold_prob"]
        elif self.local_cold_start_estimation == ColdStartEstimation.NAIVE:
            # Same prob for every function
            node_compl = sum([stats.node2completions[(_f,self.node)] for _f in self.simulation.functions])
            node_cs = sum([stats.cold_starts[(_f,self.node)] for _f in self.simulation.functions])
            for f in self.simulation.functions:
                if node_compl > 0:
                    self.cold_start_prob[(f, self.node)] = node_cs / node_compl
                else:
                    self.cold_start_prob[(f, self.node)] = COLD_START_PROB_INITIAL_GUESS
        elif self.local_cold_start_estimation == ColdStartEstimation.NAIVE_PER_FUNCTION:
            for f in self.simulation.functions:
                if stats.node2completions.get((f,self.node), 0) > 0:
                    self.cold_start_prob[(f, self.node)] = stats.cold_starts.get((f,self.node),0) / stats.node2completions.get((f,self.node),0)
                else:
                    self.cold_start_prob[(f, self.node)] = COLD_START_PROB_INITIAL_GUESS
        elif self.local_cold_start_estimation == ColdStartEstimation.NO: 
            for f in self.simulation.functions:
                self.cold_start_prob[(f, self.node)] = 0

        # CLOUD
        #
        if self.cloud_cold_start_estimation == ColdStartEstimation.PACS:
            for f in self.simulation.functions:
                total_arrival_rate = max(0.001, sum([stats.arrivals.get((f,x,self.cloud), 0.0) for x in self.simulation.classes])/self.simulation.t)
                props1, _ = perfmodel.get_sls_warm_count_dist(total_arrival_rate,
                                                            self.estimated_service_time[f],
                                                            self.estimated_service_time[f] + self.simulation.init_time[(f,self.cloud)],
                                                            self.simulation.expiration_timeout)
                self.cold_start_prob[(f, self.cloud)] = props1["cold_prob"]
        elif self.cloud_cold_start_estimation == ColdStartEstimation.NAIVE:
            # Same prob for every function
            node_compl = sum([stats.node2completions[(_f,self.cloud)] for _f in self.simulation.functions])
            node_cs = sum([stats.cold_starts[(_f,self.cloud)] for _f in self.simulation.functions])
            for f in self.simulation.functions:
                if node_compl > 0:
                    self.cold_start_prob[(f, self.cloud)] = node_cs / node_compl
                else:
                    self.cold_start_prob[(f, self.cloud)] = COLD_START_PROB_INITIAL_GUESS
        elif self.cloud_cold_start_estimation == ColdStartEstimation.NAIVE_PER_FUNCTION:
            for f in self.simulation.functions:
                if stats.node2completions.get((f,self.cloud), 0) > 0:
                    self.cold_start_prob[(f, self.cloud)] = stats.cold_starts.get((f,self.cloud),0) / stats.node2completions.get((f,self.cloud),0)
                else:
                    self.cold_start_prob[(f, self.cloud)] = COLD_START_PROB_INITIAL_GUESS
        elif self.cloud_cold_start_estimation == ColdStartEstimation.NO: 
            for f in self.simulation.functions:
                self.cold_start_prob[(f, self.cloud)] = 0

    def update(self):
        stats = self.simulation.stats

        for f in self.simulation.functions:
            if stats.node2completions[(f, self.node)] > 0:
                self.estimated_service_time[f] = stats.execution_time_sum[(f, self.node)] / \
                                                 stats.node2completions[(f, self.node)]
            else:
                self.estimated_service_time[f] = 0.1
            if stats.node2completions[(f, self.cloud)] > 0:
                self.estimated_service_time_cloud[f] = stats.execution_time_sum[(f, self.cloud)] / \
                                                       stats.node2completions[(f, self.cloud)]
            else:
                self.estimated_service_time_cloud[f] = 0.1

        self.update_cold_start(stats)


class GreedyBudgetAware(GreedyPolicy):

    def __init__ (self, simulation, node):
        super().__init__(simulation, node)

    def schedule(self, f, c, offloaded_from):
        latency_local, latency_cloud = self._estimate_latency(f,c)
        local_ok = self.can_execute_locally(f)
        budget_ok = self.simulation.stats.cost / self.simulation.t * 3600 < self.budget

        if not budget_ok and not local_ok:
            return (SchedulerDecision.DROP, None)
        if local_ok and latency_local < latency_cloud:
            return (SchedulerDecision.EXEC, None)
        if budget_ok:
            return (SchedulerDecision.OFFLOAD_CLOUD, self.cloud)
        else:
            return (SchedulerDecision.EXEC, None)


class GreedyPolicyWithCostMinimization(GreedyPolicy):

    def __init__ (self, simulation, node):
        super().__init__(simulation, node)
        # Pick the closest cloud node
        nodes_w_lat = [(_n,simulation.infra.get_latency(node,_n)) for _n in simulation.infra.get_cloud_nodes()]
        self.cloud = sorted(nodes_w_lat, key=lambda x: x[1])[0][0]

    def schedule(self, f, c, offloaded_from):
        if self.local_cold_start_estimation == ColdStartEstimation.FULL_KNOWLEDGE:
            if f in self.node.warm_pool:
                self.cold_start_prob[(f, self.node)] = 0
            else:
                self.cold_start_prob[(f, self.node)] = 1
        if self.cloud_cold_start_estimation == ColdStartEstimation.FULL_KNOWLEDGE:
            if f in self.cloud.warm_pool:
                self.cold_start_prob[(f, self.cloud)] = 0
            else:
                self.cold_start_prob[(f, self.cloud)] = 1

        latency_local = self.estimated_service_time.get(f, 0) + \
                        self.cold_start_prob.get((f, self.node), 1) * \
                        self.simulation.init_time[(f,self.node)]

        latency_cloud = self.estimated_service_time_cloud.get(f, 0) + 2 * self.simulation.infra.get_latency(self.node, self.cloud) + \
                        self.cold_start_prob.get((f, self.cloud), 1) * self.simulation.init_time[(f,self.cloud)] +\
                        f.inputSizeMean*8/1000/1000/self.simulation.infra.get_bandwidth(self.node, self.cloud)

        if latency_local < c.max_rt and self.can_execute_locally(f):
            # Choose the configuration with minimum cost (edge execution) if both configuration can execute within
            # the deadline
            sched_decision = SchedulerDecision.EXEC, None
        elif latency_cloud < c.max_rt:
            sched_decision = SchedulerDecision.OFFLOAD_CLOUD, self.cloud
        elif self.can_execute_locally(f):
            sched_decision = SchedulerDecision.EXEC, None
        else:
            sched_decision = SchedulerDecision.OFFLOAD_CLOUD, self.cloud

        return sched_decision


# LOAD BALANCER: Random  
class RandomLBPolicy(Policy):
    def __init__(self, simulation, node):
        super().__init__(simulation, node)
        self.rng = self.simulation.random_lb_rng
        print("[RandomPolicy]: Random policy is active")

    def schedule(self, f, c, offloaded_from):
        nodes = self.simulation.infra.get_cloud_nodes()
        # print("[Random]: available could nodes -> ", nodes)
        selected_node = self.rng.choice(nodes)
        # print("[Random]: selected cloud node -> ", nodes.index(selected_node))
        return (SchedulerDecision.OFFLOAD_CLOUD, selected_node)


# LOAD BALANCER: Round-Robin
class RoundRobinLBPolicy(Policy):
    def __init__(self, simulation, node):
        super().__init__(simulation, node)
        # Index to keep track of round robin server selection
        self.round_robin_index = 0
        print("[RoundRobin]: Round Robin policy is active")

    def schedule(self, f, c, offloaded_from):
        nodes = self.simulation.infra.get_cloud_nodes()
        #print("[RoundRobin]: available cloud nodes -> ", nodes)
        node_index = self.round_robin_index % len(nodes)
        # To avoid potential overflows
        self.round_robin_index = node_index
        self.round_robin_index += 1
        #print("[RoundRobin]: selected cloud node -> ", node_index)
        return (SchedulerDecision.OFFLOAD_CLOUD, nodes[node_index])


# LOAD BALANCER: MA/MA (Sophon: max_mem_available either for warm and cold start)
class MAMALBPolicy(Policy):
    def __init__(self, simulation, node):
        super().__init__(simulation, node)
        print("[MAMA]: MAMA policy is active")

    def schedule(self, f, c, offloaded_from):
        nodes = self.simulation.infra.get_cloud_nodes()
        # Prendo i container warm per quella funzione presenti sui nodi
        nodes_warm = []
        for n in nodes:
            if f in n.warm_pool:
                nodes_warm.append(n)
        if len(nodes_warm) == 0: # Nessun container warm disponibile (cold start)
            node = self._get_node_with_max_available_mem(nodes)
        else: # Container warm disponibili (warm start)
            node = self._get_node_with_max_available_mem(nodes_warm)
        #print("selected_node: ", node.name)
        return (SchedulerDecision.OFFLOAD_CLOUD, node)
    
    def _get_node_with_max_available_mem(self, nodes):
        node = nodes[0]       
        for n in nodes:
            if n.curr_memory > node.curr_memory:
                node = n
        return node

"""
# LOAD BALANCER: Weighted Round Robin (speedup, memory)
class WeightedRoundRobinLBPolicy(Policy):
    def __init__(self, simulation, node):
        super().__init__(simulation, node)
        self.MULT_FACTOR = 10
        self.ALPHA = 0.3
        self.BETA = 0.7
        self.GAMMA = 0.3
        # Index to keep track of round robin server selection
        self.round_robin_index = 0
        self.counter = 0
        self.max_speedup = 0
        self.max_memory = 0
        self.max_cost = 0
        self.node2weight = []
        self._init()
        print("[WRR]: Weighted Round Robin policy is active")
  
    def schedule(self, f, c, offloaded_from):
        nodes = self.simulation.infra.get_cloud_nodes()
        node_index = self.round_robin_index % len(nodes)
        self.counter += 1
        if self.counter >= list(self.node2weight[node_index].values())[0]:
            self.counter = 0
            # To avoid potential overflows
            self.round_robin_index = node_index
            self.round_robin_index += 1
        return (SchedulerDecision.OFFLOAD_CLOUD, nodes[node_index])

    def _init(self):
        nodes = self.simulation.infra.get_cloud_nodes()
        self.max_speedup = nodes[0].speedup
        self.max_memory = nodes[0].total_memory
        self.max_cost = nodes[0].cost
        # Recupera lo speedup, la memoria e il costo massimi
        for n in nodes:
            self.max_speedup = max(n.speedup, self.max_speedup)
            self.max_memory = max(n.total_memory, self.max_memory)
            self.max_cost = max(n.cost, self.max_cost)
        # Calcola punteggi
        for n in nodes:
            self.node2weight.append({n:self._get_weight(n)})
        self.node2weight = sorted(self.node2weight, key=lambda x: list(x.values())[0], reverse=False)

    def _get_weight(self, node: Node):
        node_weight = int(self.MULT_FACTOR * \
            (self.ALPHA * (node.speedup / self.max_speedup) \
            + self.BETA * (node.total_memory / self.max_memory)))
        return node_weight
"""

# LOAD BALANCER: Weighted Round Robin (speedup, memory)
class WeightedRoundRobinLBPolicy(Policy):
    def __init__(self, simulation, node):
        super().__init__(simulation, node)
        self.MULT_FACTOR = 10
        # Index to keep track of round robin server selection
        self.round_robin_index = 0
        self.counter = 0
        self.node2weight = {}
        self.node2count = {}
  
    def schedule(self, f, c, offloaded_from):
        nodes = self.simulation.infra.get_cloud_nodes()
        node_index = self.round_robin_index % len(nodes)
        i = 0
        while self.node2count[nodes[node_index]] == 0:
            # To avoid potential overflows
            self.round_robin_index = node_index
            self.round_robin_index += 1
            node_index = self.round_robin_index % len(nodes)
            i += 1
            if i == len(nodes):
                # reset self.node2count
                self.node2count = self.node2weight.copy()
                node_index = 0
        self.node2count[nodes[node_index]] -= 1
        self.round_robin_index = node_index
        self.round_robin_index += 1
        #print("[RoundRobin]: selected cloud node -> ", node_index)
        return (SchedulerDecision.OFFLOAD_CLOUD, nodes[node_index])


# LOAD BALANCER: WRR-speedup
class WRRSpeedupLBPolicy(WeightedRoundRobinLBPolicy):
    def __init__(self, simulation, node):
        super().__init__(simulation, node)
        self.max_speedup = 0
        self._init()
        print("[WRRSpeedup]: WRRSpeedup is active")
    
    def schedule(self, f, c, offloaded_from):
        return super().schedule(f, c, offloaded_from)

    def _init(self):
        nodes = self.simulation.infra.get_cloud_nodes()
        self.max_speedup = nodes[0].speedup
        # Recupera lo speedup massimo
        for n in nodes:
            self.max_speedup = max(n.speedup, self.max_speedup)
        # Calcola punteggi
        for n in nodes:
            weight = self._get_weight(n)
            if weight < 1:
                weight = 1
            self.node2weight[n] = weight
            self.node2count = self.node2weight.copy()

    def _get_weight(self, node: Node):
        node_weight = int(self.MULT_FACTOR * (node.speedup / self.max_speedup))
        return node_weight
    

# LOAD BALANCER: WRR-memory
class WRRMemoryLBPolicy(WeightedRoundRobinLBPolicy):
    def __init__(self, simulation, node):
        super().__init__(simulation, node)
        self.max_memory = 0
        self._init()
        print("[WRRMemory]: WRRMemory is active")
    
    def schedule(self, f, c, offloaded_from):
        return super().schedule(f, c, offloaded_from)

    def _init(self):
        nodes = self.simulation.infra.get_cloud_nodes()
        self.max_memory = nodes[0].total_memory
        # Recupera la memoria massima
        for n in nodes:
            self.max_memory = max(n.total_memory, self.max_memory)
        # Calcola punteggi
        for n in nodes:
            weight = self._get_weight(n)
            if weight < 1:
                weight = 1
            self.node2weight[n] = weight
            self.node2count = self.node2weight.copy()

    def _get_weight(self, node: Node):
        node_weight = int(self.MULT_FACTOR * (node.total_memory / self.max_memory))
        return node_weight
    

# LOAD BALANCER: WRR-cost
class WRRCostLBPolicy(WeightedRoundRobinLBPolicy):
    def __init__(self, simulation, node):
        super().__init__(simulation, node)
        self.min_cost = 0
        self._init()
        print("[WRRCost]: WRRCost is active")
    
    def schedule(self, f, c, offloaded_from):
        return super().schedule(f, c, offloaded_from)

    def _init(self):
        nodes = self.simulation.infra.get_cloud_nodes()
        self.min_cost = nodes[0].cost
        # Recupera il costo minimo
        for n in nodes:
            self.min_cost = min(n.cost, self.min_cost)
        # Calcola punteggi
        for n in nodes:
            weight = self._get_weight(n)
            if weight < 1:
                weight = 1
            self.node2weight[n] = weight
            self.node2count = self.node2weight.copy()

    def _get_weight(self, node: Node):
        node_weight = int(self.MULT_FACTOR * (self.min_cost / node.cost))
        return node_weight
    

# LOAD BALANCER: Consistent Hashing & Memory Available
class ConsistentHashingLBPolicy(Policy):
    def __init__(self, simulation, node):
        super().__init__(simulation, node)
        self.nodes = self.simulation.infra.get_cloud_nodes()
        self.ring = []
        for node in self.nodes:
            self._add_node(node)
        print("[CH]: Consistent Hashing policy is active")

    def schedule(self, f, c, offloaded_from):
        node = self._get_node(f)
        #print("selected_node: ", node)
        return (SchedulerDecision.OFFLOAD_CLOUD, node)
    
    def _hash(self, key):
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)

    def _add_node(self, node):
        key = self._hash(str(node.name))
        self.ring.append((key, node))
        self.ring.sort()
    
    def _get_node(self, f):
        key = self._hash(f.name)
        start_index = bisect.bisect_right(self.ring, (key,))
        for _, node in self.ring[start_index:] + self.ring[:start_index]:
            if (f in node.warm_pool) or (node.curr_memory >= f.memory):
                return node
        return self.nodes[start_index]
    