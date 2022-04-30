import networkx as nx
from math import ceil
from itertools import tee, product
from typing import List, Dict
from collections import defaultdict
from functools import reduce
import pdb

def pairwise(iterable):
    """
    A pairwise iterator. Source:
    https://docs.python.org/3.8/library/itertools.html#itertools-recipes
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class Stop(object):
    """
    A stop object, to be used in our simulations. Encodes a single stop.
    """
    def __init__(self, position, time, stop_type, req_id):
        self.position = position
        self.time = time
        self.stop_type = stop_type
        self.req_id = req_id
    def __repr__(self):
        return f"Stop(position={self.position}, time={self.time}, "\
               f"stop_type={self.stop_type}, req_id={self.req_id})"

class Request(object):
    """
    A request object, to be used in our simulations. Encodes a single request.
    """
    def __init__(self, req_id, req_epoch, origin, destination):
        self.req_id = req_id
        self.req_epoch = req_epoch
        self.origin = origin
        self.destination = destination
    def __repr__(self):
        return f"Request(req_id={self.req_id}, req_epoch={self.req_epoch}, "\
               f"origin={self.origin}, destination={self.destination})"

class Network(object):
    """
    A wrapper around nx.Graph. The idea is to  would override key methods
    to make simulations faster.
    """
    def __init__(self, G, network_type):
        """
        Args:
        -----
            G: Either a networkx.Graph or a Network object.
            network_type: A string. Used for smart route volume
                computations. If set to 'novolcomp', no route volume
                computation is performed.
        """
        self.network_type = network_type
        if isinstance(G, Network):
            # If a Network is passed, just copy relevant stuff.
            self._network = G._network
            self._all_shortest_paths = G._all_shortest_paths
            self._all_shortest_path_lengths = G._all_shortest_path_lengths
        else:
            self._network = nx.Graph(G)
            # Cache all shortest paths
            self._all_shortest_paths = dict(nx.all_pairs_shortest_path(self._network))
            self._all_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(self._network))

    def shortest_path_length(self, u,v, **kwargs):
        return self._all_shortest_path_lengths[u][v]

    def shortest_path(self, u,v, **kwargs):
        return self._all_shortest_paths[u][v]

    def stuff_enroute(self, s, t):
        """
        Returns all the nodes that are "enroute" when one goes from
        s to t. Basically this is the set of all nodes in all the
        shortest paths between s and t.
        """
        if self.network_type in ('ring', 'line'):
            return set(self.shortest_path(s,t))
        elif self.network_type == 'grid':
            # In a grid, we need the rectangle with corners given by
            # s and t
            s_x, s_y = s
            t_x, t_y = t

            xmin, xmax = min(s_x, t_x), max(s_x, t_x)
            ymin, ymax = min(s_y, t_y), max(s_y, t_y)
            return product(range(xmin, xmax+1), range(ymin, ymax+1))

        elif self.network_type == 'star':
            return set(self.shortest_path(s,t))
        elif self.network_type == 'novolcomp':
            # forcibly disable volume computation
            return set()
        else:
            # Do brute-force computation
            return {no for pa in nx.all_shortest_paths(
                self._network, s, t) for no in pa}

class ZeroDetourBus(object):
    """
    A simulator that simulates a single bus with no-detour policy.
    Any network can be chosen.
    """
    def __init__(self, network, req_gen, network_type, initpos=None):
        self.network_type = network_type
        self.network: Network = Network(network,
                network_type=self.network_type)
        self.req_gen = req_gen
        self.initpos = initpos

        self.position = self.initpos
        self.time = 0
        self.remaining_time = 0
        self.next_stop = None

        self.stoplist: List[Stop] = []

        # req_data contains for each request:
        # req_epoch, origin, destination, pickup_epoch, dropoff-epoch
        self.req_data = dict()
        # insertion_data contains for each insertion
        # time, stoplist_length, stoplist_volume, rest_stoplist_volume,
        # pickup_idx, dropoff_idx, insertion_type
        # insertion_type is:
        #  1 if both PU and DO en route
        #  2 if only PU en route
        #  3 if neither PU nor DO en route
        self.insertion_data_columns = ('time', 'stoplist_length', 'stoplist_volume',
                                       'rest_stoplist_volume',
                                       'pickup_index', 'dropoff_index', 'insertion_type', 'pickup_enroute',
                                       'dropoff_enroute')
        self.insertion_data = []

    def process_new_request(self, req: Request):
        """
        Process a a new request. Before doing that, fast-forward internal clock
        to request epoch and serve all requests pending until that time.
        """
        # process all requests till now and advance clock
        self.fast_forward(req.req_epoch)
        # insert into stoplist
        self.insert_req(req)

    def fast_forward(self, t):
        """
        Service all stops till a time t.
        Set self.position to the correct value.

        We need to be careful since we may be in between two
        nodes.

        There are a few points to be careful about.
        1. Are we exactly at a node, or moving along an edge?
        - This is relatively straightforward. If we are *not* at a node,
          (self.time, self.position) specifies when and where we will be after
          the edge is traversed. So we can just pretend the time is a bit
          higher than it actually is and we are actually at a node.
        2. Are we currently idle?
        - Signified by stoplist = []. We need do nothing except advance clock.
        3. Will we be idle when we have fast forwared till t?
        - Signified by all stops getting serviced.
        """
        # TODO: Possible bug
        # We may need t <= self.time here.
        # otherwise, we append dummy stop with time *larger* than
        # the first stop in the list, if a race condition occurs.
        if t < self.time:
            # We are still "in the middle of an edge".
            # There can't be any need to process stops.
            assert(self.time - t) <= 1
            self.remaining_time -= self.time - t
            dummy_stop = Stop(position=self.position,
                              time=self.time,
                              stop_type=-1,
                              req_id=None)
            self.stoplist.insert(0, dummy_stop)
        else:
            # we have crossed the next node already. i.e. we are not in the middle
            # of an edge.
            self.remaining_time = 0
            idx = 0
            for idx, s in enumerate(self.stoplist):
                self.next_stop = s
                if s.time <= t: # t or self.time?
                    self._process_stop(s)
                else:
                    break
            else:
                idx = len(self.stoplist) # this is crucial: if all the stops have been
                # processed (serviced) we must truncate the stoplist at the end

            # at this point self.time and self.position is stuck to the
            # last serviced stop. We need to now interpolate.
            # Note: even if no stop needed to be serviced, if's alright
            # self.next_stop = next stop on route
            # self.time = next node after the jump ends
            # self.remaining_time = 0
            if len(self.stoplist) > 0:
                self.stoplist = self.stoplist[idx:] # is there an
                                                    # indexing error? No.
                # else: stoplist already empty. idx not yet defined.
                # no need to prune the stoplist
            if len(self.stoplist) > 0: # we are *not* idling
                self.position, remaining_time = self.interpolate(
                        curtime=t, started_from=self.position,
                        going_to=self.next_stop.position,
                        started_at=self.time
                        )
                self.time = t + remaining_time # this would put t (potentially) to the
                # future. This is alright. See the beginning of this function.
                self.remaining_time = remaining_time
            else:
                # We are idle now. self.position should be correct. Just set
                # self.time
                self.time = t
            dummy_stop = Stop(position=self.position,
                              time=self.time,
                              stop_type=-1,
                              req_id=None)
            self.stoplist.insert(0, dummy_stop)

    def insert_req(self, req: Request):
        """
        Inserts req to self.stoplist
        Logs details of the request and the insertion.
        """
        # check if pickup is on the route
        pickup_enroute = False
        dropoff_enroute = False
        assert self.stoplist[0].stop_type == -1

        def _insert_stop_at_middle(idx, position, arrtime, stop_type):
            stop = Stop(position=position,
                        time=arrtime,
                        stop_type=stop_type,
                        req_id=req.req_id)
            self.stoplist.insert(idx, stop)

        def _insert_stop_at_end(position, arrtime, stop_type):
            stop = Stop(position=position,
                       time=arrtime,
                       stop_type=stop_type,
                       req_id=req.req_id)
            self.stoplist.append(stop)

        vol = len(reduce(set.union,
                             (self.network.stuff_enroute(u.position,v.position) for u,v in
                              pairwise(self.stoplist)), set()))
        for idx, (u,v) in enumerate(pairwise(self.stoplist)):

            is_inbetween, dist_to, dist_from, dist_direct \
                = self._is_between(req.origin, u.position, v.position)
            if is_inbetween:
                pickup_enroute = True
                pickup_idx = idx+1
                pickup_epoch = u.time+dist_to
                _insert_stop_at_middle(
                    pickup_idx, req.origin, arrtime=pickup_epoch, stop_type=1)
                vol_rest = len(reduce(set.union,
                             (self.network.stuff_enroute(u.position,v.position) for u,v in
                              pairwise(self.stoplist[pickup_idx:])), set())) - 1
                # then check if dropoff is on the route
                for idx, (w,x) in enumerate(pairwise(self.stoplist[pickup_idx:])):
                    is_inbetween, dist_to, dist_from, dist_direct \
                        = self._is_between(req.destination, w.position, x.position)
                    if is_inbetween:
                        dropoff_enroute = True
                        dropoff_epoch = w.time+dist_to
                        dropoff_idx = pickup_idx+idx+1 # TODO: is this +1 necessary?
                        _insert_stop_at_middle(
                            dropoff_idx, req.destination,
                            arrtime=dropoff_epoch, stop_type=0)
                        self.insertion_data.append(
                            (self.time, len(self.stoplist)-2, vol, vol_rest, pickup_idx,
                             dropoff_idx, 1, pickup_enroute, dropoff_enroute))
                        break
                else:
                    dropoff_enroute = False
                    # append dropoff at the end
                    dist_to = self.network.shortest_path_length(
                        self.stoplist[-1].position, req.destination)
                    dropoff_epoch = self.stoplist[-1].time+dist_to
                    _insert_stop_at_end(req.destination,  dropoff_epoch, 0)
                    dropoff_idx = len(self.stoplist)-2
                    self.insertion_data.append(
                        (self.time, len(self.stoplist)-2, vol, vol_rest, pickup_idx,\
                         len(self.stoplist)-2, 2, pickup_enroute, dropoff_enroute))
                break
        else:
            vol_rest = 0
            pickup_enroute = False
            dropoff_enroute = False
            # append PU/DO at the end
            pos_last_stop = self.stoplist[-1].position
            time_last_stop = self.stoplist[-1].time
            # first, pickup
            dist_to = self.network.shortest_path_length(pos_last_stop, req.origin)
            pickup_epoch = time_last_stop+dist_to
            _insert_stop_at_end(req.origin, pickup_epoch, 1)
            # then, dropoff
            dist_to = self.network.shortest_path_length(
                self.stoplist[-1].position, req.destination)
            dropoff_epoch = self.stoplist[-1].time+dist_to
            _insert_stop_at_end(req.destination, dropoff_epoch, 0)

            dropoff_idx = len(self.stoplist)-2
            pickup_idx = len(self.stoplist)-2
            self.insertion_data.append(
                (self.time, len(self.stoplist)-2, vol, vol_rest, len(self.stoplist)-2,
                 len(self.stoplist)-2, 3, pickup_enroute, dropoff_enroute))

        # store self.req_data
        self.req_data[req.req_id] = dict(origin=req.origin,
                                         destination=req.destination,
                                         req_epoch=req.req_epoch, # NOT self.time
                                         # since jump. see function fast_forward
                                         pickup_epoch=pickup_epoch,
                                         dropoff_epoch=dropoff_epoch
                                         )
        dummy_stop = self.stoplist.pop(0) # remove dummy stop
        assert dummy_stop.stop_type == -1

    def simulate_all_requests(self):
        """
        simulates the system till req_gen is empty
        """
        for req in self.req_gen:
            self.process_new_request(req)
        print(f"simulation complete. current time {self.time}")

    def _process_stop(self, s: Stop):
        self.time = s.time
        self.remaining_time = 0
        self.position = s.position

        if s.stop_type == 1:
            assert self.req_data[s.req_id]['pickup_epoch'] == self.time
        else:
            assert s.stop_type == 0
            assert self.req_data[s.req_id]['dropoff_epoch'] == self.time

    def interpolate(self, curtime, started_from, going_to, started_at):
        """
        Returns:
        --------
            position, remaining_time: position is the next node on the way, remaining_time
                is the remaining time necessary to reach position.
        """
        if curtime == started_at:
            return started_from, 0
        if curtime < started_at:
            raise ValueError(f"curtime ({curtime}) cannot be smaller than"
                             f"started_at ({started_at})")
        shortest_path = self.network.shortest_path(started_from, going_to)
        shortest_path_length = len(shortest_path) - 1

        if curtime >= started_at + shortest_path_length:
            return going_to, 0

        delta_t = curtime - started_at
        num_nodes_traversed = ceil(delta_t) # next node
        remaining_time = num_nodes_traversed - delta_t
        pos = shortest_path[num_nodes_traversed]
        return pos, remaining_time

    def _is_between(self, a, u, v):
        """
        checks if a is on a shortest path between u and v
        """
        dist_to = self.network.shortest_path_length(u, a)
        dist_from = self.network.shortest_path_length(a, v)
        dist_direct = self.network.shortest_path_length(u, v)
        is_inbetween = dist_to + dist_from == dist_direct
        if is_inbetween:
            return True, dist_to, dist_from, dist_direct
        else:
            return False, dist_to, dist_from, dist_direct
