"""
RideRequest class
"""


class RideRequest:
    def __init__(self, start, end, time_req, path, bid, share_ride):
        self.start = start
        self.end = end
        self.time_req = time_req
        self.path = path
        self.bid = bid
        self.share_ride = share_ride


