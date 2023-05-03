"""
Authors: Elisabetta Perotti and Ana Ospina
Date: 5/2/23
Case 1. Considers EV fleet.
Both ride and charge requests are sent to the ride-servive provider.
Produce avg results over a fixed number of iterations.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pulp
import tripData as tData
import functions as f
import time
import cvxpy as cp

# Imported classes
from Vehicle import Vehicle
from ChargeRequest import ChargeRequest
from RideRequest import RideRequest

np.random.seed(10)

# Fleet size
n_veh = 100

# Number of iterations for avg random SOC case
num_iter = 10

# PV generation profile, select data to import
pv_prof = tData.pv_sunny
# pv_prof = tData.pv_cloud_am
# pv_prof = tData.pv_cloud_pm
charge_req_prob = np.array(pv_prof)

# Initialize arrays
num_charge_stations = 4
miss_ride_time_avg = [0]*tData.num_min
y_power_cars_avg = [[0]*num_charge_stations for i in range(tData.num_min)]
high_battery_time_avg = [0]*tData.num_min
int_battery_time_avg = [0]*tData.num_min
low_battery_time_avg = [0]*tData.num_min
incent_charge_assigned_avg = [0] * tData.num_min
incent_ride_assigned_avg = [0] * tData.num_min

ev_ride_time_avg = [0] * tData.num_min
ev_charge_time_avg = [0] * tData.num_min
ev_idle_time_avg = [0] * tData.num_min

# Variable initialization
delta_ride = 2  # Max zones away from customer for pick-up
delta_charge = 1  # Max zones away from charging station
min_travel_time = 5  # min
min_consume = 0.5  # kWh
station_power = 25  # kW per station at peak power
power_transferred = 12  # kW for each car charging
discharge_rate = 0.1  # kWh/minute
charge_rate = discharge_rate * 2
infeasib_cost = 1e5
infeasib_threas = 1e4
travel_edge_time = 10
in_value = 1.0 / n_veh  # for initialization of x_ij

# random SOC array
min_charge = 6
intermediate_charge = 31
full_charge = 50  # kWh
random_soc = [[np.random.uniform(min_charge, full_charge) for n in range(n_veh)] for iter in range(num_iter)]
random_noise = [np.random.binomial(1, charge_req_prob[minu]) for minu in range(24*60)]

# DiGraph - unweighted
g = nx.DiGraph()
elist = [(1, 2), (1, 3), (2, 1), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 4), (3, 5), (4, 2), (4, 3), (4, 5),
         (4, 6),
         (4, 7), (5, 2), (5, 3), (5, 4),
         (5, 6), (5, 7), (6, 4), (6, 5), (6, 7), (6, 8), (6, 9), (7, 4), (7, 6), (7, 8), (7, 9), (8, 6), (8, 7),
         (8, 9),
         (9, 6), (9, 7), (9, 8)]
g.add_edges_from(elist)
n = g.number_of_nodes()

for it in range(num_iter):
    np.random.seed(0)
    print("*** Iteration: " + str(it) + " ***")
    # Reset vehicle positions
    vehicles = []
    for j in range(n_veh // n):
        vehicles.extend([Vehicle(pos) for pos in range(1, n + 1)])
    for j in range(n_veh - len(vehicles)):
        vehicles.append(Vehicle(np.random.randint(1, n + 1)))

    # Initiate the state of charge for each vehicle
    for j in range(n_veh):
        vehicles[j].stateOfCharge = random_soc[it][j]

    # Track Pref
    y_Pref_current = []
    y_power_cars = []
    station_nodes = [3, 5, 8, 9]  # nodes where charging facilities exist
    cars_charging = [0] * len(station_nodes)
    future_cars_charging = [0] * len(station_nodes)

    # Variables to track performance
    miss_ride_time = []
    miss_charge_time = []
    low_battery_time = []
    int_battery_time = []
    high_battery_time = []
    incent_ride_assigned = []
    incent_charge_assigned = []

    ev_ride_time = []
    ev_charge_time = []
    ev_idle_time = []

    h_format = []

    # Iterate over requests, deltaT = 1 min
    for k in range(tData.num_min):
        PULoc = tData.records[k][0]
        DOLoc = tData.records[k][1]
        h_bid = [np.random.uniform(tData.h_aux_min[k], tData.h_aux[k]) for i in range(5)]
        b_min = [np.random.uniform(-tData.b_aux[k], 0) for i in range(len(station_nodes))]
        b_max = [np.random.uniform(0, tData.b_aux[k]) for i in range(len(station_nodes))]
        c_RES = tData.c_RES_total[k]
        alpha_w = tData.alpha_w_total[k]
        minute = k + tData.h_in * 60
        numRideReq = len(PULoc)
        h_format.append(time.strftime("%H:%M", time.gmtime(minute * 60)))

        print("*** Minute: " + str(k) + " ***")

        req_vec = []  # List of requests collected between time t and t+1
        req_idx = 0
        ride_req_idx = 0
        charge_req_idx = 0

        start_costs = [0] * n_veh  # costs to get to the pick-up point
        start_path = [0] * n_veh  # path to get to the pick-up point

        start_costs_charge = [0] * n_veh  # costs to get to charging station
        start_path_charge = [0] * n_veh  # path to get to the charging station

        cost = []
        paths = []

        # Update vehicle positions
        for j in range(n_veh):
            # If vehicle reaches passenger final destination
            if vehicles[j].request is not None and isinstance(vehicles[j].request, RideRequest) and vehicles[j].estimated_done_riding <= k:
                vehicles[j].position = vehicles[j].request.end
                vehicles[j].request = None  # Vehicle again available
            # If vehicle reaches charging station
            if vehicles[j].request is not None and isinstance(vehicles[j].request, ChargeRequest):
                if vehicles[j].estimated_done_riding <= k and not vehicles[j].charging:
                    vehicles[j].position = vehicles[j].request.location
                    vehicles[j].charging = True
                    cars_charging[station_nodes.index(vehicles[j].position)] += 1
                elif vehicles[j].estimated_done_charging - min_travel_time <= k and vehicles[j].charging and not \
                vehicles[j].removed:
                    future_cars_charging[station_nodes.index(vehicles[j].request.location)] -= 1
                    vehicles[j].removed = True


        # Update vehicle state-of-charge
        for j in range(n_veh):
            if vehicles[j].estimated_done_riding >= k:
                vehicles[j].stateOfCharge -= discharge_rate
            elif vehicles[j].charging:
                vehicles[j].stateOfCharge += charge_rate
                # If vehicle is fully charged disconnect
                if vehicles[j].stateOfCharge >= full_charge:
                    vehicles[j].stateOfCharge = full_charge
                    vehicles[j].charging = False
                    cars_charging[station_nodes.index(vehicles[j].request.location)] -= 1
                    vehicles[j].request = None  # Vehicle again available

            if vehicles[j].stateOfCharge < 0:
                raise Exception("ERROR: negative SOC")

        outofcharge_cars = 0
        intcharge_cars = 0
        highcharge_cars = 0
        for v in vehicles:
            if v.stateOfCharge < min_charge:
                outofcharge_cars += 1
            elif v.stateOfCharge < intermediate_charge:
                intcharge_cars += 1
            else:
                highcharge_cars += 1
        low_battery_time.append(outofcharge_cars)
        int_battery_time.append(intcharge_cars)
        high_battery_time.append(highcharge_cars)

        ride_cars = 0
        charge_cars = 0
        for v in vehicles:
            if isinstance(v.request, RideRequest):
                ride_cars += 1
            elif isinstance(v.request, ChargeRequest):
                charge_cars += 1
        ev_ride_time.append(ride_cars)
        ev_charge_time.append(charge_cars)
        ev_idle_time.append(n_veh - (ride_cars + charge_cars))

        # Generate a ride request
        for i in range(numRideReq):
            ride_req = f.create_ride_request(PULoc[i], DOLoc[i], k, g, h_bid[i], 0)
            req_vec.insert(req_idx, ride_req)

            # Compute cost of reaching the pickup point for each vehicle
            for j in range(n_veh):
                current_pos = vehicles[j].position
                start_path[j] = nx.shortest_path(g, source=current_pos,
                                                 target=ride_req.start)  # Path from current vehicle node to start node
                if vehicles[j].request is not None or \
                        vehicles[j].stateOfCharge < min_consume + len(start_path[j]) + len(req_vec[req_idx].path) - 2 or \
                        vehicles[j].stateOfCharge < min_charge or \
                        delta_ride < len(start_path[j]) - 1:
                    start_costs[j] = infeasib_cost
                else:
                    start_costs[j] = len(start_path[j]) - 1

            cost.append(start_costs.copy())
            paths.append(start_path.copy())
            ride_req_idx += 1
            req_idx += 1

        # Generate one or more charge requests
        PrefMax = np.array([12, 19, 6, 2]) * station_power
        y_Pref_current.append(PrefMax * charge_req_prob[minute])

        PrefAvailable = []
        for i in range(len(station_nodes)):
            PrefAvailable.append(round(charge_req_prob[minute] * PrefMax[i] - power_transferred * future_cars_charging[i]))

        Pref = 0
        num_req_station = []
        for p in PrefAvailable:
            if p >= full_charge // 2 and random_noise[minute]:
                Pref += p
                charge_loc = station_nodes[PrefAvailable.index(p)]
                numChargeReq = p // (full_charge // 2)
                num_req_station.append(numChargeReq)
                for i in range(numChargeReq):
                    charge_req = f.create_charge_request(charge_loc, k)
                    req_vec.insert(req_idx, charge_req)
                    charge_req_idx += 1
                    req_idx += 1

                    # Compute cost of reaching the charging station point for each vehicle
                    for j in range(n_veh):
                        current_pos = vehicles[j].position
                        start_path_charge[j] = nx.shortest_path(g, source=current_pos,
                                                                target=charge_req.location)  # Path from current vehicle node to charging station node
                        if vehicles[j].request is not None or vehicles[j].stateOfCharge > 2 / 3 * full_charge or \
                                vehicles[j].stateOfCharge < min_consume + len(start_path_charge[j]) - 1 or \
                                delta_charge < len(start_path_charge[j]) - 1:
                            start_costs_charge[j] = infeasib_cost
                        else:
                            start_costs_charge[j] = len(start_path_charge[j]) - 1

                    cost.append(start_costs_charge.copy())
                    paths.append(start_path_charge.copy())
            else:
                num_req_station.append(0)

        if not req_idx:
            print("No real request at this time!")
            y_power_cars.append(np.array(cars_charging) * power_transferred)
            miss_ride_time.append(0)
            miss_charge_time.append(0)
            incent_ride_assigned.append(0)
            incent_charge_assigned.append(0)
            continue
        elif sum([sum(cost[i]) for i in range(req_idx)]) == infeasib_cost * n_veh * req_idx:
            print("WARNING: No feasible request at this time!")
            y_power_cars.append(np.array(cars_charging) * power_transferred)
            incent_ride_assigned.append(0)
            incent_charge_assigned.append(0)
            if ride_req_idx:
                miss_ride_time.append(ride_req_idx)
            else:
                miss_ride_time.append(0)
            if charge_req_idx:
                miss_charge_time.append(charge_req_idx)
            else:
                miss_charge_time.append(0)
            continue

        # Add virtual requests or vehicles to make the assignment matrix square
        n_assign = max(req_idx, n_veh)
        if req_idx < n_veh:
            virtual_req_costs = [[infeasib_cost for col in range(n_veh)] for row in range(n_veh - req_idx)]
            cost = np.vstack([cost, virtual_req_costs]).transpose()
        elif req_idx > n_veh:
            virtual_req_costs = [[infeasib_cost for col in range(req_idx)] for row in range(req_idx - n_veh)]
            cost = np.vstack([np.array(cost).transpose(), virtual_req_costs])
        else:
            cost = np.array(cost).transpose()

        cost_array = cost.transpose().flatten()
        inf_idx = [i for i, x in enumerate(cost_array) if x == infeasib_cost]
        feas_idx_num = n_assign * n_assign - len(inf_idx)

        # Solve assignment problem
        Kout = 2  # Iterations outer loop
        Kin = 50  # Iterations inner loop

        yStar = [0] * n_assign * n_assign

        skip = 0
        for kreq in range(ride_req_idx):
            for kk in range(n_veh):
                yStar[kk + (n_veh + skip) * kreq] = req_vec[kreq].bid - alpha_w * (
                            len(req_vec[kreq].path) + len(paths[kreq][kk]) - 2)
            if n_veh < req_idx:
                skip = req_idx - n_veh

        for kreq in range(ride_req_idx, req_idx):
            for kk in range(n_veh):
                yStar[kk + (n_veh + skip) * kreq] = 0.01 * (full_charge - vehicles[kk].stateOfCharge)
            if n_veh < req_idx:
                skip = req_idx - n_veh

        y = [[0 for col in range(Kout)] for row in range(n_assign * n_assign)]
        xi = [[0 for col in range(Kout + 1)] for row in range(n_assign * n_assign)]
        # Initialize with xi inside the feasible set, i.e. satisfies the constraints.
        for i in range(n_assign * n_assign):
            xi[i][0] = in_value

        cost_function = np.zeros(n_assign * n_assign)
        yStar_feas_ride = []
        for jj in range(ride_req_idx):  # Loop over requests - columns
            for ii in range(n_veh):  # Loop over vehicles - rows
                req_j = ii * (1 + jj) + (n_assign - ii) * jj
                if req_j not in inf_idx:
                    yStar_feas_ride.append(yStar[req_j])

        yStar_feas_charge = []
        for s in range(len(station_nodes)):
            yStar_feas_ch_s = []
            ind_in = sum(num_req_station[0:s], ride_req_idx)
            ind_fin = sum(num_req_station[0:s + 1], ride_req_idx)
            for jj in range(ind_in, ind_fin):  # Loop over charge requests at station s - columns
                for ii in range(n_veh):  # Loop over vehicles - rows
                    req_j = ii * (1 + jj) + (n_assign - ii) * jj
                    if req_j not in inf_idx:
                        yStar_feas_ch_s.append(yStar[req_j])
            yStar_feas_charge.append(yStar_feas_ch_s)

        for t in range(Kout):
            x_feas_ride = []
            for jj in range(ride_req_idx):  # Loop over ride requests - columns
                for ii in range(n_veh):  # Loop over vehicles - rows
                    req_j = ii * (1 + jj) + (n_assign - ii) * jj
                    if req_j not in inf_idx:
                        x_feas_ride.append(xi[req_j][t])

            x_feas_ch = []
            for s in range(len(station_nodes)):
                x_feas_ch_s = []
                ind_in = sum(num_req_station[0:s], ride_req_idx)
                ind_fin = sum(num_req_station[0:s + 1], ride_req_idx)
                for jj in range(ind_in, ind_fin):  # Loop over charge requests at station s - columns
                    for ii in range(n_veh):  # Loop over vehicles - rows
                        req_j = ii * (1 + jj) + (n_assign - ii) * jj
                        if req_j not in inf_idx:
                            x_feas_ch_s.append(xi[req_j][t])
                x_feas_ch.append(x_feas_ch_s)

            # Solve inner using cvxpy
            fin_incent_y = []
            # ride req part
            if len(x_feas_ride):
                yinn_ride = cp.Variable(len(x_feas_ride))
                objective = cp.Minimize(cp.sum_squares(yinn_ride - np.array(yStar_feas_ride)))
                constraint = [-1 <= yinn_ride, yinn_ride <= 1]
                problem = cp.Problem(objective, constraint)
                problem.solve()
                fin_incent_y = yinn_ride.value.tolist()

                if t == Kout - 1:
                    if np.sum(x_feas_ride) != 0:
                        incent_ride_assigned.append(
                            np.dot(np.array(x_feas_ride), np.array(fin_incent_y)) / np.sum(x_feas_ride))
                    else:
                        incent_ride_assigned.append(0)
            else:
                if t == Kout - 1:
                    incent_ride_assigned.append(0)

            # charge req part
            sum_charge_inc = 0
            for s in range(len(station_nodes)):
                if len(x_feas_ch[s]):
                    yinn_ch = cp.Variable(len(x_feas_ch[s]))
                    obj = cp.Minimize((PrefAvailable[s] * c_RES - yinn_ch @ np.array(x_feas_ch[s]) ** 2))
                    constr = [yinn_ch @ np.array(x_feas_ch[s]) >= b_min[s],
                              yinn_ch @ np.array(x_feas_ch[s]) <= b_max[s], -1 <= yinn_ch, yinn_ch <= 1]
                    probl = cp.Problem(obj, constr)
                    probl.solve()
                    fin_incent_y = np.concatenate((fin_incent_y, yinn_ch.value)).tolist()

                    if t == Kout - 1:
                        if np.sum(x_feas_ch[s]) != 0:
                            sum_charge_inc += np.dot(np.array(x_feas_ch[s]), np.array(yinn_ch.value)) / np.sum(
                                x_feas_ch[s])

            if t == Kout - 1:
                incent_charge_assigned.append(sum_charge_inc)

            # Cost function for the assignment problem - based on original cost and price incentives
            for jj in range(n_assign):  # Loop over requests - columns
                for ii in range(n_assign):  # Loop over vehicles - rows
                    req_j = ii * (1 + jj) + (n_assign - ii) * jj
                    if req_j not in inf_idx:
                        cost_function[req_j] = cost[ii][jj] - fin_incent_y.pop(0)
                    else:
                        cost_function[req_j] = cost[ii][jj]

            if fin_incent_y:
                raise Exception("ERROR: incorrect number of incentives")

            # Solve the assignment problem using PuLP, a Python toolbox
            prob = pulp.LpProblem("AssignmentProblem", pulp.LpMinimize)

            x_list = []
            for i in range(n_assign * n_assign):
                x_list.append(pulp.LpVariable("x" + str(i), 0, 1, pulp.LpInteger))

            prob += pulp.lpSum(cost_function[m] * x_list[m] for m in range(n_assign * n_assign)), "obj function"

            for i in range(n_assign):
                prob += pulp.lpSum(x_list[i * n_assign + n] for n in range(n_assign)) == 1, "c_eq" + str(i)  # sum column elements
                prob += pulp.lpSum(x_list[i + n_assign * n] for n in range(n_assign)) <= 1, "c_ineq" + str(i)  # sum row elements

            prob.solve(pulp.GLPK(msg=0))

            for v in prob.variables():
                index = int(v.name[1:])
                xi[index][t + 1] = v.varValue

        x_fin = np.array(xi)[:, -1]

        X = np.array(x_fin).reshape(n_assign, n_assign).transpose()
        cost_mat = cost_function.reshape(n_assign, n_assign).transpose()
        C_X = X * cost_mat

        # Assign requests
        for i in range(n_assign):
            veh_idx = X[:, i].tolist().index(1)
            if C_X[veh_idx][i] > infeasib_threas:  # Skip unfeasible requests
                continue
            vehicles[veh_idx].request = req_vec[i]  # Assign request to vehicle
            if isinstance(req_vec[i], RideRequest):
                paths[i][veh_idx].pop()
                vehicles[veh_idx].path = paths[i][veh_idx] + req_vec[i].path  # Assign path (pickup + ride)
                vehicles[veh_idx].estimated_done_riding = k + (len(vehicles[veh_idx].path) - 1) * travel_edge_time + min_travel_time
            elif isinstance(req_vec[i], ChargeRequest):
                vehicles[veh_idx].path = paths[i][veh_idx]  # Assign path to reach the charging station
                ai = full_charge - vehicles[veh_idx].stateOfCharge + len(
                    vehicles[veh_idx].path) - 1 + 0.5  # car discharges even more to reach the station
                vehicles[veh_idx].estimated_done_riding = k + (
                        len(vehicles[veh_idx].path) - 1) * travel_edge_time + min_travel_time
                vehicles[veh_idx].estimated_done_charging = vehicles[veh_idx].estimated_done_riding + int(
                    round(ai / power_transferred * 60))
                future_cars_charging[station_nodes.index(vehicles[veh_idx].request.location)] += 1
                vehicles[veh_idx].removed = False
            else:
                print("Wrong request assignment")

        y_power_cars.append(np.array(cars_charging) * power_transferred)

        count_assigned_rides = 0
        count_assigned_charges = 0
        for i in vehicles:
            if i.request in req_vec and isinstance(i.request, RideRequest):
                count_assigned_rides += 1
            elif i.request in req_vec and isinstance(i.request, ChargeRequest):
                count_assigned_charges += 1
        if count_assigned_rides == ride_req_idx:
            miss_ride_time.append(0)
        else:
            miss_ride_time.append(ride_req_idx - count_assigned_rides)

        if count_assigned_charges == charge_req_idx:
            miss_charge_time.append(0)
        else:
            miss_charge_time.append(charge_req_idx - count_assigned_charges)

    # Test
    for i in range(len(low_battery_time)):
        if not low_battery_time[i] + int_battery_time[i] + high_battery_time[i] == n_veh:
            raise Exception("ERROR: wrong soc estimate")

    # Sum data
    miss_ride_time_avg += np.array(miss_ride_time)
    y_power_cars_avg += np.array(y_power_cars)
    high_battery_time_avg += np.array(high_battery_time)
    int_battery_time_avg += np.array(int_battery_time)
    low_battery_time_avg += np.array(low_battery_time)
    incent_charge_assigned_avg += np.array(incent_charge_assigned)
    incent_ride_assigned_avg += np.array(incent_ride_assigned)
    ev_ride_time_avg += np.array(ev_ride_time)
    ev_charge_time_avg += np.array(ev_charge_time)
    ev_idle_time_avg += np.array(ev_idle_time)

# Average data, no rounding yet
y_power_cars_avg = y_power_cars_avg/num_iter
miss_ride_time_avg = miss_ride_time_avg/num_iter
high_battery_time_avg = high_battery_time_avg/num_iter
int_battery_time_avg = int_battery_time_avg/num_iter
low_battery_time_avg = low_battery_time_avg/num_iter
incent_charge_assigned_avg = incent_charge_assigned_avg / num_iter
incent_ride_assigned_avg = incent_ride_assigned_avg / num_iter
ev_ride_time_avg = ev_ride_time_avg / num_iter
ev_charge_time_avg = ev_charge_time_avg / num_iter
ev_idle_time_avg = ev_idle_time_avg / num_iter

# Re-name variables
miss_ride_time = miss_ride_time_avg
y_power_cars = y_power_cars_avg
high_battery_time = high_battery_time_avg
int_battery_time = int_battery_time_avg
low_battery_time = low_battery_time_avg
incent_charge_assigned = incent_charge_assigned_avg
incent_ride_assigned = incent_ride_assigned_avg
ev_ride_time = ev_ride_time_avg
ev_charge_time = ev_charge_time_avg
ev_idle_time = ev_idle_time_avg

high_battery_time = np.rint(high_battery_time)
int_battery_time = np.rint(int_battery_time)
low_battery_time = np.rint(low_battery_time)
ev_ride_time = np.rint(ev_ride_time)
ev_charge_time = np.rint(ev_charge_time)
ev_idle_time = np.rint(ev_idle_time)

time_slot = 15 # 15-min time slots

print("  *** Avg results, rounded *** ")
print("  --- Vehicles with low battery: " + str(low_battery_time[-1]))
print("  --- Vehicles with int battery: " + str(int_battery_time[-1]))
print("  --- Vehicles with high battery: " + str(high_battery_time[-1]))
print("Missed ride-req, sum min by min: " + str(sum(miss_ride_time)))
print("Missed ride-req, rounded 5-min sum: " + str(int(sum(np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1))))))

lost_power_percent = 0
for i in range(len(y_Pref_current)):
    if sum(y_Pref_current[i]) > sum(y_power_cars[i]):
        lost_power_percent += (sum(y_Pref_current[i]) - sum(y_power_cars[i]))

tot_Pref = sum([sum(i) for i in y_Pref_current])
print("Power lost: " + str(round((lost_power_percent / tot_Pref)*100, 2)) + "%")
print("QoS: " + str(100-(int(sum(np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1)))))/2462*100))


# Plots
# Pref and missing rides
plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 14})
plt.rcParams['axes.xmargin'] = 0

fig, host = plt.subplots(figsize=(5, 4), layout="constrained") # layout='constrained')  # (width, height) in inches
ax3 = host.twinx()
host.set_xlabel("Time")
host.set_ylabel("Power [kW]")
ax3.set_ylabel("Missed ride requests")
ax3.locator_params(axis="y", integer=True, tight=True)
p1 = host.plot(h_format, [sum(i) for i in y_Pref_current], label='$P_{\mathrm{ref}}$', color="#1EBFE1")
p1b = host.plot(h_format, [sum(i) for i in y_power_cars], label='$v_{\mathrm{ch}} p_{\mathrm{ch}}$', color="#1E5EE1")
p3 = ax3.bar(h_format[::time_slot], np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1)), width=7, alpha=0.3, color = "#C8377E", label=("Total missed ride requests: " + str(int(sum(np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1)))))))
ax3.legend(loc='upper center', fontsize="14", ncol=1, bbox_to_anchor=(0.5, 1.15))
host.legend(handles=p1+p1b, loc='upper right', fontsize="14", ncol=1)
host.set_xticks(h_format[::360])
host.set_xticklabels(h_format[::360])
plt.savefig("BiSu_PrefMissRide_avg.pdf", bbox_inches='tight')

#SOC
plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 12})
plt.rcParams['axes.xmargin'] = 0

fig3, ax3 = plt.subplots(figsize=(4,2.5), tight_layout=True)
ax3.bar(h_format, high_battery_time, width=1.0, alpha=0.5, label="High SOC", color = "limegreen")
ax3.bar(h_format, int_battery_time, width=1.0, alpha=0.5, label="Mid SOC", color = "gold")
ax3.bar(h_format, low_battery_time, width=1.0, alpha=0.6, label="Low SOC", color = "orangered")
ax3.set_xticks(h_format[::360])
ax3.set_xticklabels(h_format[::360])
ax3.set_xlabel('Time')
ax3.set_ylabel('Number of EV')
ax3.margins(y=0)
ax3.legend(loc='upper center', fontsize="10",  bbox_to_anchor=(0.44, 1.25), ncol=3)
fig3.savefig('BiSu_SOC_avg.pdf', bbox_inches='tight')

# EV request status
plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 12})
plt.rcParams['axes.xmargin'] = 0

fig7, ax7 = plt.subplots(figsize=(4,2.5), tight_layout=True)
ax7.bar(h_format, ev_idle_time, width=1.0, alpha=0.6, label="Idling", color = "#C53AA6")
ax7.bar(h_format, ev_ride_time, width=1.0, alpha=0.5, label="Riding", color = "#3AA6C5")
ax7.bar(h_format, ev_charge_time, width=1.0, alpha=0.5, label="Charging", color = "#A6C53A")
ax7.set_xticks(h_format[::360])
ax7.set_xticklabels(h_format[::360])
ax7.set_xlabel('Time')
ax7.set_ylabel('Number of EV')
ax7.margins(y=0)
ax7.legend(loc='upper center', fontsize="10", bbox_to_anchor=(0.5, 1.3), ncol=3)
fig7.savefig('BiSu_EV_avg.pdf', bbox_inches='tight')

# Pref and charging profile at each station
plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 7.5})
plt.rcParams['axes.xmargin'] = 0

fig4, axs = plt.subplots(1, 4, figsize=(8,1.5), tight_layout=True)
for i in range(4):
    plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 7.5})
    plt.rcParams['axes.xmargin'] = 0
    axs[i].plot(h_format, [y_Pref_current[k][i] for k in range(len(y_Pref_current))], label='$P_{\mathrm{ref}}$', color="#1EBFE1")
    axs[i].plot(h_format, [y_power_cars[k][i] for k in range(len(y_power_cars))], label='$v_{\mathrm{ch}} p_{\mathrm{ch}}$', color="#1E5EE1")
    axs[i].set_xticks(h_format[::360])
    axs[i].set_xticklabels(h_format[::360])
    axs[i].set_xlabel('Time')
axs[0].set_ylabel('Power [kW]')
axs[3].legend(loc='upper right', ncol=1)
fig4.savefig('BiSu_power4stations_avg.pdf', bbox_inches='tight')

# Incentives assigned
#plt.style.use('seaborn-deep')
plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 11})
plt.rcParams['axes.xmargin'] = 0

fig5, ax5 = plt.subplots(figsize=(4, 2), tight_layout=True)
ax5.bar(h_format[::time_slot],
        np.sum(np.array(incent_charge_assigned).reshape(len(incent_charge_assigned) // time_slot, time_slot),
               axis=1), width=0.5, alpha=0.5, label="Charge request", color="#33397E")
ax5.bar(h_format[::time_slot],
        np.sum(np.array(incent_ride_assigned).reshape(len(incent_ride_assigned) // time_slot, time_slot), axis=1),
        width=0.5, alpha=0.5, label="Ride request", color="#7e7833")
ax5.set_xticks(h_format[::360])
ax5.set_xticklabels(h_format[::360])
ax5.set_xlabel('Time')
ax5.set_ylabel('Incentive[$\$$]')
ax5.margins(y=0)
ax5.set_ylim(-1, 11)
ax5.legend(loc='upper center', fontsize="10", ncol=2)
fig5.savefig('BiSu_incAssign_avg.pdf', bbox_inches='tight')

plt.show()
# %%
