import networkx as nx

# Import classes
from RideRequest import RideRequest
from ChargeRequest import ChargeRequest

# Functions
def create_ride_request(start, end, time, graph, bid, share_ride):
    start = start
    end = end
    time_req = time  # Time at which the request is placed
    path = nx.shortest_path(graph, source=start, target=end)
    bid = bid
    share_ride = share_ride
    return RideRequest(start, end, time_req, path, bid, share_ride)

def create_charge_request(location, time):
    location = location
    time_req = time
    return ChargeRequest(location, time_req)
