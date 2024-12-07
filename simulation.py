import configparser
from dataclasses import dataclass, field
import time
from heapq import heappop, heappush
import numpy as np
from numpy.random import SeedSequence, default_rng
import sys

import conf
import utils.plot # type: ignore
from policy import SchedulerDecision
import policy
import probabilistic
from faas import *
import stateful
from stateful import key_locator
from arrivals import ArrivalProcess
from infrastructure import *
from statistics import Stats # type: ignore
from typing import List

from mab import EpsilonGreedy, UCB, ResetUCB, SlidingWindowUCB, UCB2, UCBTuned, KLUCB

@dataclass
class Event:
    canceled: bool = field(default=False, init=False)

    # XXX: ugly workaround to avoid issues with the heapq (in case of events
    # scheduled at the same time)
    def __lt__(self, other):
        return True

    def __le__(self,other):
        return True


@dataclass
class Arrival(Event):
    node: Node
    function: Function
    qos_class: QoSClass
    arrival_proc: ArrivalProcess = None
    offloaded_from: [Node] = field(default_factory=list) # type: ignore
    original_arrival_time: float = None


@dataclass
class CheckExpiredContainers(Event):
    node: Node


@dataclass
class PolicyUpdate(Event):
    pass


@dataclass
class ArrivalRateUpdate(Event):
    pass


@dataclass
class StatPrinter(Event):
    pass


@dataclass
class Completion(Event):
    arrival: float
    function: Function
    qos_class: QoSClass
    node: Node
    cold: bool
    exec_time: float
    offloaded_from: [Node] = None # type: ignore
    data_access_time: float = 0.0


@dataclass
class MABUpdate(Event):
    pass

@dataclass
class RewardUpdate(Event):
    pass

@dataclass
class RewardConfig:
    alpha: float
    beta: float
    gamma: float
    delta: float
    zeta: float

    # non-stationary context
    alpha_post:float
    beta_post:float
    gamma_post:float
    delta_post:float
    zeta_post:float


OFFLOADING_OVERHEAD = 0.005
ARRIVAL_TRACE_PERIOD = 60.0


@dataclass
class Simulation:

    config: configparser.ConfigParser
    seed_sequence: SeedSequence
    infra: Infrastructure
    functions: [Function] # type: ignore
    classes: [QoSClass] # type: ignore
    node2arrivals: dict

    def __post_init__ (self):
        assert(len(self.functions) > 0)
        assert(len(self.classes) > 0)

        self.__event_counter = 0

        self.stats = Stats(self, self.functions, self.classes, self.infra)

        self.first_stat_print = True
        self.external_arrivals_allowed = True

        self.verbosity = self.config.getint(conf.SEC_SIM, conf.VERBOSITY, fallback=0)

        # Seeds
        n_arrival_processes = sum([len(arrival_procs) for arrival_procs in self.node2arrivals.values()])
        # Spawn off child SeedSequences to pass to child processes.
        child_seeds = self.seed_sequence.spawn(8 + 3*n_arrival_processes)
        self.service_rng = default_rng(child_seeds[0])
        self.node_choice_rng = default_rng(child_seeds[1])
        self.policy_rng1 = default_rng(child_seeds[2])
        self.keys_rng = default_rng(child_seeds[3])
        self.keys_policy_rng = default_rng(child_seeds[4])

        # For LOAD BALANCER
        self.random_lb_rng = default_rng(child_seeds[5])
        self.wrr_lb_rng = default_rng(child_seeds[6])
        self.mab_agent_rng = default_rng(child_seeds[7])

        i = 8
        for n,arvs in self.node2arrivals.items():
            for arv in arvs:
                arv.init_rng(default_rng(child_seeds[i]), default_rng(child_seeds[i+1]), default_rng(child_seeds[i+2]))
                i += 3

        self.max_neighbors = self.config.getint(conf.SEC_SIM, conf.EDGE_NEIGHBORS, fallback=3)

        # Other params
        self.init_time = {}
        for node in self.infra.get_nodes():
            if node.speedup > 0:
                for fun in self.functions:
                    self.init_time[(fun,node)] = fun.initMean/node.speedup

    def new_policy (self, configured_policy, node):
        if configured_policy == "basic":
            return policy.BasicPolicy(self, node)
        if configured_policy == "basic-budget":
            return policy.BasicBudgetAwarePolicy(self, node)
        if configured_policy == "basic-edge":
            return policy.BasicEdgePolicy(self, node)
        if configured_policy == "cloud":
            return policy.CloudPolicy(self, node)
        elif configured_policy == "probabilistic-legacy":
            self.config.set(conf.SEC_POLICY, conf.EDGE_OFFLOADING_ENABLED, "false")
            return probabilistic.ProbabilisticPolicy(self, node)
        elif configured_policy == "probabilistic-legacy-strict":
            self.config.set(conf.SEC_POLICY, conf.EDGE_OFFLOADING_ENABLED, "false")
            return probabilistic.ProbabilisticPolicy(self, node, True)
        elif configured_policy == "probabilistic":
            return probabilistic.ProbabilisticPolicy(self, node)
        elif configured_policy == "probabilistic-strict":
            return probabilistic.ProbabilisticPolicy(self, node, True)
        elif configured_policy == "probabilistic-strictAlt" or configured_policy == "fgcs24":
            self.config.set(conf.SEC_POLICY, conf.MULTIPLE_OFFLOADING_ALLOWED, "false")
            return probabilistic.ProbabilisticPolicy(self, node, True)
        elif configured_policy == "probabilistic-offline":
            return probabilistic.OfflineProbabilisticPolicy(self, node)
        elif configured_policy == "probabilistic-offline-strict":
            return probabilistic.OfflineProbabilisticPolicy(self, node, True)
        elif configured_policy == "greedy":
            return policy.GreedyPolicy(self, node)
        elif configured_policy == "greedy-budget":
            return policy.GreedyBudgetAware(self, node)
        elif configured_policy == "greedy-min-cost":
            return policy.GreedyPolicyWithCostMinimization(self, node)
        elif configured_policy == "random":
            return probabilistic.RandomPolicy(self, node)
        elif configured_policy == "random-stateful":
            return stateful.RandomStatefulOffloadingPolicy(self, node)
        elif configured_policy == "state-aware":
            return stateful.StateAwareOffloadingPolicy(self, node)
        elif configured_policy == "state-aware-always-offload":
            return stateful.AlwaysOffloadStatefulPolicy(self, node)
        elif configured_policy == "random-lb": # load balancer
            return policy.RandomLBPolicy(self, node)
        elif configured_policy == "round-robin-lb": # load balancer
            return policy.RoundRobinLBPolicy(self, node)
        elif configured_policy == "mama-lb": # load balancer
            return policy.MAMALBPolicy(self, node)
        elif configured_policy == "weighted-round-robin-lb": # load balancer
            return policy.WeightedRoundRobinLBPolicy(self, node)
        elif configured_policy == "wrr-speedup-lb": # load balancer
            return policy.WRRSpeedupLBPolicy(self, node)
        elif configured_policy == "wrr-memory-lb": # load balancer
            return policy.WRRMemoryLBPolicy(self, node)
        elif configured_policy == "wrr-cost-lb": # load balancer
            return policy.WRRCostLBPolicy(self, node)
        elif configured_policy == "const-hash-lb": # load balancer
            return policy.ConsistentHashingLBPolicy(self, node)
        else:
            raise RuntimeError(f"Unknown policy: {configured_policy}")

    def new_state_migration_policy(self, stateful_policy_name: str):
        if stateful_policy_name == "none":
            return None
        elif stateful_policy_name == "random":
            return stateful.RandomKeyMigrationPolicy(self, self.keys_policy_rng)
        elif stateful_policy_name == "gradient-discent":
            return stateful.GradientBasedMigrationPolicy(self, self.keys_policy_rng)
        elif stateful_policy_name == "spring-based":
            return stateful.SpringBasedMigrationPolicy(self, self.keys_policy_rng)
        elif stateful_policy_name == "greedy":
            return stateful.SimpleGreedyMigrationPolicy(self, self.keys_policy_rng)
        elif stateful_policy_name == "ilp-min-access":
            return stateful.ILPMinDataAccessTimeMigrationPolicy(self, self.keys_policy_rng)
        elif stateful_policy_name == "ilp":
            return stateful.ILPBoundedDataAccessTimeMigrationPolicy(self, self.keys_policy_rng)
        else:
            raise RuntimeError(f"Unknown state migration policy: {stateful_policy_name}")

    def new_mab_agent(self):
        lb_policies = self.config.get(conf.SEC_MAB, conf.MAB_LB_POLICIES, fallback=[]).split(', ')
        strategy = self.config.get(conf.SEC_MAB, conf.MAB_STRATEGY, fallback="Epsilon-Greedy")
        reward_config = RewardConfig(
            alpha=self.config.getfloat(conf.SEC_MAB, conf.MAB_REWARD_ALPHA, fallback=0.0), 
            beta=self.config.getfloat(conf.SEC_MAB, conf.MAB_REWARD_BETA, fallback=0.0), 
            gamma=self.config.getfloat(conf.SEC_MAB, conf.MAB_REWARD_GAMMA, fallback=0.0), 
            delta=self.config.getfloat(conf.SEC_MAB, conf.MAB_REWARD_DELTA, fallback=0.0),
            zeta=self.config.getfloat(conf.SEC_MAB, conf.MAB_REWARD_ZETA, fallback=0.0),

            # non-stationary
            alpha_post=self.config.getfloat(conf.SEC_MAB, conf.MAB_REWARD_ALPHA_POST, fallback=0.0),
            beta_post=self.config.getfloat(conf.SEC_MAB, conf.MAB_REWARD_BETA_POST, fallback=0.0),
            gamma_post=self.config.getfloat(conf.SEC_MAB, conf.MAB_REWARD_GAMMA_POST, fallback=0.0),
            delta_post=self.config.getfloat(conf.SEC_MAB, conf.MAB_REWARD_DELTA_POST, fallback=0.0),
            zeta_post=self.config.getfloat(conf.SEC_MAB, conf.MAB_REWARD_ZETA_POST, fallback=0.0)
        )
        if strategy == "Epsilon-Greedy":
            epsilon = self.config.getfloat(conf.SEC_MAB, conf.MAB_EPSILON, fallback=0.1)
            return EpsilonGreedy(self, lb_policies, epsilon, reward_config)
        elif strategy == "UCB":
            exploration_factor = self.config.getfloat(conf.SEC_MAB, conf.MAB_UCB_EXPLORATION_FACTOR, fallback=0.05)
            return UCB(self, lb_policies, exploration_factor, reward_config)
        elif strategy == "ResetUCB":
            exploration_factor = self.config.getfloat(conf.SEC_MAB, conf.MAB_UCB_EXPLORATION_FACTOR, fallback=0.05)
            reset_interval = self.config.getint(conf.SEC_MAB, conf.MAB_RUCB_INTERVAL, fallback=sys.maxsize)
            return ResetUCB(self, lb_policies, exploration_factor, reset_interval, reward_config)
        elif strategy == "SlidingWindowUCB":
            exploration_factor = self.config.getfloat(conf.SEC_MAB, conf.MAB_UCB_EXPLORATION_FACTOR, fallback=0.05)
            window_size = self.config.getint(conf.SEC_MAB, conf.MAB_SWUCB_WINDOW_SIZE, fallback=10)
            return SlidingWindowUCB(self, lb_policies, exploration_factor, window_size, reward_config)
        elif strategy == "UCB2":
            exploration_factor = self.config.getfloat(conf.SEC_MAB, conf.MAB_UCB_EXPLORATION_FACTOR, fallback=0.05)
            alpha = self.config.getfloat(conf.SEC_MAB, conf.MAB_UCB2_ALPHA, fallback=1.0)
            return UCB2(self, lb_policies, exploration_factor, reward_config, alpha)
        elif strategy == "UCBTuned":
            exploration_factor = self.config.getfloat(conf.SEC_MAB, conf.MAB_UCB_EXPLORATION_FACTOR, fallback=0.05)
            return UCBTuned(self, lb_policies, exploration_factor, reward_config)
        elif strategy == "KL-UCB":
            exploration_factor = self.config.getfloat(conf.SEC_MAB, conf.MAB_UCB_EXPLORATION_FACTOR, fallback=0.05)
            c = self.config.getfloat(conf.SEC_MAB, conf.MAB_KL_UCB_C, fallback=1.0)
            return KLUCB(self, lb_policies, exploration_factor, reward_config, c)
        else:
            print("Unknown MAB strategy\n")
            exit(1)

    def run (self):
        # Simulate
        self.close_the_door_time = self.config.getfloat(conf.SEC_SIM, conf.CLOSE_DOOR_TIME, fallback=100)
        self.events = []
        self.t = 0.0
        self.node2policy = {}

        self.expiration_timeout = self.config.getfloat(conf.SEC_CONTAINER, conf.EXPIRATION_TIMEOUT, fallback=600)

        # Policy
        policy_name = self.config.get(conf.SEC_POLICY, conf.POLICY_NAME, fallback="basic")
        for n in self.infra.get_edge_nodes():
            _policy = n.custom_sched_policy if n.custom_sched_policy is not None else policy_name
            self.node2policy[n] = self.new_policy(_policy, n)
        for n in self.infra.get_cloud_nodes():
            _policy = n.custom_sched_policy if n.custom_sched_policy is not None else "cloud"
            self.node2policy[n] = self.new_policy(_policy, n)
        # For LOAD BALANCER
        lb_policy = self.config.get(conf.SEC_LB, conf.LB_POLICY, fallback="random-lb")
        for n in self.infra.get_load_balancers():
            _lb_policy = n.custom_sched_policy if n.custom_sched_policy is not None else lb_policy
            self.node2policy[n] = self.new_policy(_lb_policy, n)

        self.key_migration_policy = None
        stateful_policy_name = self.config.get(conf.SEC_STATEFUL, conf.POLICY_NAME, fallback="none")
        self.key_migration_policy = self.new_state_migration_policy(stateful_policy_name)

        self.policy_update_interval = self.config.getfloat(conf.SEC_POLICY, conf.POLICY_UPDATE_INTERVAL, fallback=-1)
        self.rate_update_interval = self.config.getfloat(conf.SEC_SIM, conf.RATE_UPDATE_INTERVAL, fallback=-1)
        print(self.rate_update_interval)
        self.stats_print_interval = self.config.getfloat(conf.SEC_SIM, conf.STAT_PRINT_INTERVAL, fallback=-1)
        self.stats_file = sys.stdout
        self.mab_stats_file = sys.stdout

        # Multi-Armed Bandit
        self.mab_update_interval = self.config.getfloat(conf.SEC_MAB, conf.MAB_UPDATE_INTERVAL, fallback=-1)

        rt_print_filename = self.config.get(conf.SEC_SIM, conf.RESP_TIMES_FILE, fallback="")
        if len(rt_print_filename) > 0:
            self.resp_times_file = open(rt_print_filename, "w")
            print(f"Function,Class,Node,Offloaded,Cold,DataAccess,RT", file=self.resp_times_file)
        else:
            self.resp_times_file = None

        if not self.config.getboolean(conf.SEC_SIM, conf.PLOT_RESP_TIMES, fallback=False):
            self.resp_time_samples = {}
        else:
            self.resp_time_samples = {(f,c): [] for f in self.functions for c in f.get_invoking_classes()}

        for n, arvs in self.node2arrivals.copy().items():
            for arv in arvs:
                self.__schedule_next_arrival(n, arv)

        if len(self.events) == 0:
            # No arrivals
            print("No arrivals configured.")
            exit(1)

        # Initialize state placement
        stateful.init_key_placement (self.functions, self.infra, self.keys_rng)

        if self.policy_update_interval > 0.0:
            self.schedule(self.policy_update_interval, PolicyUpdate())
        if self.rate_update_interval > 0.0:
            self.schedule(self.rate_update_interval, ArrivalRateUpdate())
        if self.stats_print_interval > 0.0:
            self.schedule(self.stats_print_interval, StatPrinter())
            stats_print_filename = self.config.get(conf.SEC_SIM, conf.STAT_PRINT_FILE, fallback="")
            if len(stats_print_filename) > 0:
                self.stats_file = open(stats_print_filename, "w")
        if self.mab_update_interval > 0.0:
            mab_stats_print_filename = self.config.get(conf.SEC_SIM, conf.MAB_STAT_PRINT_FILE, fallback="")
            if len(mab_stats_print_filename) > 0:
                self.mab_stats_file = open(mab_stats_print_filename, "w")
            self.mab_agent = self.new_mab_agent()
            self.current_lb_policy = lb_policy
            self.schedule(self.mab_update_interval, MABUpdate())
            # Non-stationary case
            non_stationary=self.config.getboolean(conf.SEC_MAB, conf.MAB_NON_STATIONARY_ENABLED, fallback=False)
            if non_stationary:
                self.schedule(3600, RewardUpdate())

        while len(self.events) > 0:
            t,e = heappop(self.events)
            self.handle(t, e)

        # LOAD BALANCER: MAB Agent 
        if self.mab_update_interval > 0.0:
            self.mab_update(last_update=True)

        if self.stats_print_interval > 0:
            self.print_periodic_stats()
            print("]", file=self.stats_file)
            if self.stats_file != sys.stdout:
                self.stats_file.close()
                self.stats.print(sys.stdout)
        elif self.config.getboolean(conf.SEC_SIM, conf.PRINT_FINAL_STATS, fallback=True):
            self.stats.print(sys.stdout)
        else:
            print(self.stats.utility - self.stats.penalty)

        if self.resp_times_file is not None:
            self.resp_times_file.close()

        if len(self.resp_time_samples) > 0:
            plot.plot_rt_cdf(self.resp_time_samples) # type: ignore

        for n, arvs in self.node2arrivals.items():
            for arv in arvs:
                arv.close()

        return self.stats

    def move_key (self, k, src_node, dest_node):
        if src_node == dest_node:
            return
        dest_node.kv_store[k] = src_node.kv_store[k]
        del(src_node.kv_store[k])
        key_locator.update_key_location(k, dest_node)

        moved_bytes = dest_node.kv_store[k]
        self.stats.data_migrations_count += 1
        self.stats.data_migrated_bytes += moved_bytes
    
    def __schedule_next_arrival(self, node, arrival_proc):
        if not self.external_arrivals_allowed:
            return

        c = arrival_proc.next_class()
        iat = arrival_proc.next_iat()
        f = arrival_proc.function

        if iat >= 0.0 and self.t + iat < self.close_the_door_time:
            self.schedule(self.t + iat, Arrival(node,f,c, arrival_proc))
        else:
            arrival_proc.close()
            self.node2arrivals[node].remove(arrival_proc)
            if len(self.node2arrivals[node]) == 0:
                del(self.node2arrivals[node])

        if len(self.node2arrivals) == 0:
            # Little hack: remove all expiration from the event list (we do not
            # need to wait for them)
            for item in self.events:
                if isinstance(item[1], CheckExpiredContainers) \
                   or isinstance(item[1], PolicyUpdate) \
                   or isinstance(item[1], ArrivalRateUpdate) \
                   or isinstance(item[1], StatPrinter) \
                   or isinstance(item[1], MABUpdate):
                    item[1].canceled = True
            self.external_arrivals_allowed = False

    def schedule (self, t, event):
        heappush(self.events, (t, event))

    def print_periodic_stats (self):
        # print("Step: ", self.t)
        of = self.stats_file if self.stats_file is not None else sys.stdout
        if not self.first_stat_print:
            print(",", end='', file=of)
        else:
            print("[", file=of)
        self.stats.print(of)
        self.first_stat_print = False

    def handle (self, t, event):
        if event.canceled:
            return
        self.t = t
        #print(event)

        self.__event_counter += 1
        if self.__event_counter % 10000 == 0:
            #print(t)
            self.__event_counter = 0
        if isinstance(event, Arrival):
            self.handle_arrival(event)
        elif isinstance(event, Completion):
            self.handle_completion(event)
        elif isinstance(event, PolicyUpdate):
            for n,p in self.node2policy.items():
                upd_t0 = time.time()
                p.update()
                self.stats.update_policy_upd_time(n,time.time()-upd_t0)

            # Migrate keys
            if self.key_migration_policy is not None:
                upd_t0 = time.time()
                self.key_migration_policy.update_metrics()
                self.key_migration_policy.migrate()
                for p in self.node2policy.values():
                    if isinstance(p, stateful.StateAwareOffloadingPolicy):
                        p.latency_estimation_cache = {}
                self.stats.update_mig_policy_upd_time(time.time()-upd_t0)

            self.schedule(t + self.policy_update_interval, event)
        elif isinstance(event, ArrivalRateUpdate):
            for n, arvs in self.node2arrivals.copy().items():
                for arv in arvs:
                    if arv.has_dynamic_rate():
                        arv.update_dynamic_rate()
            self.schedule(t + self.rate_update_interval, event)
        elif isinstance(event, StatPrinter):
            self.print_periodic_stats()
            self.schedule(t + self.stats_print_interval, event)
        elif isinstance(event, CheckExpiredContainers):
            if len(event.node.warm_pool) == 0:
                return
            f,timeout = event.node.warm_pool.front()
            if timeout < t:
                self.stats.update_memory_usage(event.node, self.t)
                event.node.curr_memory += f.memory
                event.node.warm_pool.pool = event.node.warm_pool.pool[1:]
        elif isinstance(event, MABUpdate):
            print("[MABUpdate]: MAB agent in action...")
            self.mab_update()
            self.schedule(t + self.mab_update_interval, event)
        elif isinstance(event, RewardUpdate):
            self.reward_update()
        else:
            raise RuntimeError("")

    def handle_completion (self, event):
        rt = self.t - event.arrival
        f = event.function
        c = event.qos_class
        n = event.node
        duration = event.exec_time
        dat = event.data_access_time
        #print(f"Completed {f}-{c}: {rt}")

        # Account for the time needed to send back the result
        if event.offloaded_from != None:
            curr_node = n
            for remote_node in reversed(event.offloaded_from):
                rt += self.infra.get_latency(curr_node, remote_node)
                curr_node = remote_node

        self.stats.resp_time_sum[(f,c,n)] += rt
        if (f,c,n) in self.resp_time_samples:
            self.resp_time_samples[(f,c,n)].append(rt)
        self.stats.completions[(f,c,n)] += 1
        self.stats.node2completions[(f,n)] += 1
        self.stats.execution_time_sum[(f,n)] += duration
        self.stats.raw_utility += c.utility
        if c.max_rt <= 0.0 or rt <= c.max_rt:
            self.stats.utility += c.utility
            self.stats.utility_detail[(f,c,n)] += c.utility
        elif c.max_rt > 0.0:
            self.stats.violations[(f,c,n)] += 1
            self.stats.penalty += c.deadline_penalty

        if f.max_data_access_time is not None and dat > f.max_data_access_time:
            self.stats.data_access_violations[f] += 1
            self.stats.data_access_tardiness += dat - f.max_data_access_time

        if n.cost > 0.0:
            self.stats.cost += duration * f.memory/1024 * n.cost

        if self.resp_times_file is not None:
            print(f"{f},{c},{n},{event.offloaded_from != None and len(event.offloaded_from) > 0},{event.cold},{dat},{rt}", file=self.resp_times_file)

        n.warm_pool.append((f, self.t + self.expiration_timeout))
        if self.external_arrivals_allowed:
            self.schedule(self.t + self.expiration_timeout, CheckExpiredContainers(n)) 

    def do_offload (self, arrival, target_node):
        latency = self.infra.get_latency(arrival.node, target_node)
        transfer_time = arrival.function.inputSizeMean*8/1000/1000/self.infra.get_bandwidth(arrival.node, target_node)
        remote_arv = Arrival(target_node, arrival.function, arrival.qos_class, offloaded_from=arrival.offloaded_from.copy())
        remote_arv.offloaded_from.append(arrival.node)
        remote_arv.original_arrival_time = self.t

        self.schedule(self.t + latency + OFFLOADING_OVERHEAD + transfer_time, remote_arv)

    def handle_arrival (self, event):
        n = event.node 
        node_policy = self.node2policy[n]
        external = len(event.offloaded_from) == 0
        arv_proc = event.arrival_proc
        f = event.function
        c = event.qos_class
        self.stats.arrivals[(f,c,n)] += 1
        if external:
            self.stats.ext_arrivals[(f,c,n)] += 1

        # For LOAD BALANCER
        if n.total_memory != 0:
            self.stats.cloud_arrivals[(f,c,n)] += 1

        # Policy
        sched_decision, target_node = node_policy.schedule(f,c,event.offloaded_from)

        if sched_decision == SchedulerDecision.EXEC:
            duration, data_access_time = self.next_function_duration(f, n)
            # check warm or cold
            if f in n.warm_pool:
                n.warm_pool.remove(f)
                init_time = 0
            else:
                self.stats.update_memory_usage(event.node, self.t)
                assert(n.curr_memory >= f.memory)
                n.curr_memory -= f.memory
                self.stats.cold_starts[(f,n)] += 1
                init_time = self.init_time[(f,n)]
            arrival_time = self.t if event.original_arrival_time is None else event.original_arrival_time
            self.schedule(float(self.t + init_time + duration), Completion(arrival_time, f,c, n, init_time > 0, duration, event.offloaded_from, data_access_time))
        elif sched_decision == SchedulerDecision.DROP:
            self.stats.dropped_reqs[(f,c,n)] += 1
            self.stats.penalty += c.drop_penalty
            if event.offloaded_from is not None and len(event.offloaded_from) > 0:
                self.stats.dropped_offloaded[(f,c,n)] += 1
        elif sched_decision == SchedulerDecision.OFFLOAD_CLOUD:
            if target_node is not None:
                remote_node = target_node
            else:
                # Pick the closest cloud node
                nodes_w_lat = [(_n,self.infra.get_latency(n,_n)) for _n in self.infra.get_cloud_nodes()]
                if len(nodes_w_lat) < 1:
                    remote_node = None
                else:
                    remote_node = sorted(nodes_w_lat, key=lambda x: x[1])[0][0]
            if remote_node is None:
                # drop
                self.stats.dropped_reqs[(f,c,n)] += 1
                self.stats.penalty += c.drop_penalty
            else:
                self.stats.offloaded[(f,c,n)] += 1
                self.do_offload(event, remote_node)  
        elif sched_decision == SchedulerDecision.OFFLOAD_EDGE:
            if target_node is not None:
                remote_node = target_node
            else:
                remote_node = node_policy.pick_edge_node(f,c)
            if remote_node is None:
                # drop
                self.stats.dropped_reqs[(f,c,n)] += 1
                self.stats.penalty += c.drop_penalty
            else:
                self.stats.offloaded[(f,c,n)] += 1
                self.do_offload(event, remote_node)  

        # Schedule next (if this is an external arrival)
        if external:
            self.__schedule_next_arrival(n, arv_proc)

    def next_function_duration (self, f: Function, n: Node):
        # execution time
        duration = float(self.service_rng.gamma(1.0/f.serviceSCV, f.serviceMean*f.serviceSCV/n.speedup) )
        data_access_time = 0
        # we add the time to access state
        for k,prob in f.accessed_keys:
            # check if it is accessed 
            if self.keys_rng.random() <= prob:
                # check if the key is on the node
                if k not in n.kv_store:
                    remote_node = stateful.key_locator.get_node(k)
                    assert(k in remote_node.kv_store)
                    value_size = remote_node.kv_store[k]
                    extra_latency = self.infra.get_latency(n, remote_node)*2
                    # bandwidth
                    extra_latency += value_size/(self.infra.get_bandwidth(n, remote_node)*125000)
                    data_access_time += extra_latency
                self.stats.data_access_count[(k,f,n)] += 1
        return float(duration + data_access_time), float(data_access_time)

    def mab_update(self, last_update=False):
        # Invoco l'agente per aggiornare il modello 
        # Si occuperà lui stesso dell'aggiornamento del reward
        if last_update:
            self.mab_agent.update_model(self.current_lb_policy, self.mab_stats_file,last_update=True)
        else:
            self.mab_agent.update_model(self.current_lb_policy, self.mab_stats_file,last_update=False)
        # Richiedo la nuova politica di load balancing all'agente
        # La politica scelta verrà applicata a tutti i load balancer presenti
        selected_policy = self.mab_agent.select_policy()
        if selected_policy != None:
            self.current_lb_policy = selected_policy
            _lb_policy = self.current_lb_policy
            for n in self.infra.get_load_balancers():
                self.node2policy[n] = self.new_policy(_lb_policy, n)

    def reward_update(self):
        self.mab_agent.ALPHA = self.mab_agent.ALPHA_POST  # load imbalance
        self.mab_agent.BETA  = self.mab_agent.BETA_POST  # response time
        self.mab_agent.GAMMA = self.mab_agent.GAMMA_POST  # cost
        self.mab_agent.DELTA = self.mab_agent.DELTA_POST  # utility
        self.mab_agent.ZETA  = self.mab_agent.ZETA_POST  # response time violations
