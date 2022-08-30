from qubo_helper import Qubo
from itertools import combinations, product

# VRP problem with multi-source.
# Class has informations about sources, costs, destinations, weights and capacities.
# Class provides methods to formule problem as QUBO problem.
class VRPProblem:

    # Parameters :
    # sources - list of ids of magazines
    # costs - 2d array with cost of travel between destinations.
    # capacities - list of capacities of vehicles
    # dests - list of destinations that needs to be served
    # weights - list of weights of orders
    # first_source - flag that says if we count travel between magazine and first destination to the cost
    # last_source - flag that says if we count travel between last destination and magazine to the cost
    def __init__(self, sources, costs, capacities, dests, weights,
            first_source = True, last_source = True):
        # Merging all sources into one source.
        source = 0
        weights[source] = 0
        self.source = source
        in_nearest_sources = dict()
        out_nearest_sources = dict()

        # Finding nearest source for all destinations.
        for dest in dests:
            in_nearest = sources[0]
            out_nearest = sources[0]
            for s in sources:
                costs[source][s] = 0
                costs[s][source] = 0
                if costs[s][dest] < costs[in_nearest][dest]:
                    in_nearest = s
                if costs[dest][s] < costs[dest][out_nearest]:
                    out_nearest = s
            costs[source][dest] = costs[in_nearest][dest]
            costs[dest][source] = costs[dest][out_nearest]
            in_nearest_sources[dest] = in_nearest
            out_nearest_sources[dest] = out_nearest

        self.costs = costs
        self.capacities = capacities
        self.dests = dests
        self.weights = weights
        self.in_nearest_sources = in_nearest_sources
        self.out_nearest_sources = out_nearest_sources
        self.first_source = first_source
        self.last_source = last_source
        self.sources = sources

    # Returns qubo with information about capacities.
    def get_capacity_qubo(self, capacity, start_step, final_step):
        dests = self.dests
        weights = self.weights
        cap_qubo = Qubo()

        for (d1, d2) in combinations(dests, 2):
            for (s1, s2) in combinations(range(start_step, final_step + 1), 2):
                index = ((s1, d1), (s2, d2))
                index2 = ((s1, d2), (s2, d1))
                cost = weights[d1] * weights[d2] / capacity**2
                cap_qubo.add(index, cost)
                cap_qubo.add(index2, cost)

        return cap_qubo

    # Returns qubo with information about costs between destinations.
    def get_order_qubo(self, start_step, final_step, dests, costs):
        source = self.source
        ord_qubo = Qubo()

        # Order constraints.
        for step in range(start_step, final_step):
            for dest1 in dests:
                for dest2 in dests:
                    cost = costs[dest1][dest2]
                    index = ((step, dest1), (step + 1, dest2))
                    ord_qubo.add(index, cost)

        return ord_qubo

    # Returns qubo with coded information about costs of travel from magazines to destinations.
    # Used if first_source parameter is True.
    def get_first_dest_qubo(self, start_step, dests, costs, source):
        fir_qubo = Qubo()

        for dest in dests:
            in_index = ((start_step, dest), (start_step, dest))
            in_cost = costs[source][dest]
            fir_qubo.add(in_index, in_cost)

        return fir_qubo

    # Returns qubo with coded information about costs of travel from destinations to magazines.
    # Used if last_source parameter is True.
    def get_last_dest_qubo(self, final_step, dests, costs, source):
        las_qubo = Qubo()

        for dest in dests:
            out_index = ((final_step, dest), (final_step, dest))
            out_cost = costs[dest][source]
            las_qubo.add(out_index, out_cost)

        return las_qubo

    # Returns qubo with additional constraint, that every vehicle has
    # specified number of deliveries that it need to serve.
    def get_qubo_with_partition(self, vehicle_partitions,
            only_one_const, order_const):
        limits = [(r, r) for r in vehicle_partitions]
        return self.get_qubo_with_both_limits(limits,
                only_one_const, order_const)

    # Returns qubo with additional constraint, that every vehicle has
    # specified maximum number of deliveries that it can serve.
    def get_qubo_with_limits(self, vehicle_limits,
            only_one_const, order_const):
        limits = [(0, r) for r in vehicle_limits]
        return self.get_qubo_with_both_limits(limits,
                only_one_const, order_const)

    # Returns qubo with additional constraint that every vehicle has
    # specified minimum and maximum number of deliveries that it can serve.
    # vehicles_limits - list of pairs (a, b), a <= b.
    def get_qubo_with_both_limits(self, vehicle_limits,
            only_one_const, order_const):
        steps = 0
        for (_, r) in vehicle_limits:
            steps += r

        capacities = self.capacities
        dests = self.dests
        source = self.source
        dests_with_source = dests.copy()
        dests_with_source.append(source)
        costs = self.costs
        vrp_qubo = Qubo()

        # Only one step for one destination.
        for dest in self.dests:
            vrp_qubo.add_only_one_constraint([(step, dest) for step in range(steps)], only_one_const) 

        start = 0
        for vehicle in range(len(vehicle_limits)):
            min_size = vehicle_limits[vehicle][0]
            max_size = vehicle_limits[vehicle][1]
            min_final = start + min_size - 1
            max_final = start + max_size - 1

            # First steps should have normal destinations.
            if min_size != 0:
                for step in range(start, min_final + 1):
                    vrp_qubo.add_only_one_constraint([(step, dest) for dest in dests], only_one_const) 
                ord_min_qubo = self.get_order_qubo(start, min_final, dests, costs)
                vrp_qubo.merge_with(ord_min_qubo, 1., order_const)

            # In other steps vehicles can wait in source.
            if max_size != min_size:
                for step in range(min_final + 1, max_final + 1):
                    vrp_qubo.add_only_one_constraint([(step, dest) for dest in dests_with_source], only_one_const)
                ord_max_qubo = self.get_order_qubo(min_final + 1, max_final, dests_with_source, costs)
                vrp_qubo.merge_with(ord_max_qubo, 1., order_const)

            # From min_final step to min_final + 1 step.
            if min_size != 0 and min_size != max_size:
                for dest1 in dests:
                    for dest2 in dests_with_source:
                        cost = costs[dest1][dest2]
                        index = ((min_final, dest1), (min_final + 1, dest2))
                        vrp_qubo.add(index, cost * order_const)

            # First and last destinations.
            if self.first_source:
                fir_qubo = None
                if min_size != 0:
                    fir_qubo = self.get_first_dest_qubo(start, dests, costs, source)
                else:
                    fir_qubo = self.get_first_dest_qubo(start, dests_with_source, costs, source)
                vrp_qubo.merge_with(fir_qubo, 1., order_const)
            if self.last_source:
                las_qubo = None
                if max_size != min_size:
                    las_qubo = self.get_last_dest_qubo(max_final, dests_with_source, costs, source)
                else:
                    las_qubo = self.get_last_dest_qubo(max_final, dests, costs, source)
                vrp_qubo.merge_with(las_qubo, 1., order_const)

            start = max_final + 1

        return vrp_qubo

    # Returns qubo without additional constraints.
    def get_full_qubo(self, only_one_const, order_const):
        dests = len(self.dests)
        vehicles = len(self.capacities)

        limits = [dests for _ in range(vehicles)]
        return self.get_qubo_with_limits(limits, only_one_const, order_const)
