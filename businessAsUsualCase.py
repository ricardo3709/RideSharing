"""
Authors: Elisabetta Perotti and Ana Ospina
Date: 5/2/23
Business-as-usual case. Considers EV fleet.
No charge-request are sent. EVs charge when their battery is below min_charge.
Produce avg results over a fixed number of iterations.
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pulp
import tripData as tData
import functions as f
import time

# Imported classes
from Vehicle import Vehicle
from RideRequest import RideRequest

np.random.seed(10)

# Fleet size
n_veh = 100

# Number of iterations for avg random SOC case
num_iter = 1

# Initialize arrays
miss_ride_time_avg = [0]*tData.num_min
y_power_cars_avg = [0]*tData.num_min
high_battery_time_avg = [0]*tData.num_min
int_battery_time_avg = [0]*tData.num_min
low_battery_time_avg = [0]*tData.num_min
ev_ride_time_avg = [0] * tData.num_min
ev_charge_time_avg = [0] * tData.num_min
ev_idle_time_avg = [0] * tData.num_min

# Variable initialization
delta_ride = 2  # Max zones away from customer for pick-up
min_travel_time = 5  # min
min_consume = 0.5  # kWh
power_transferred = 12  # kW for each car charging

# discharge_rate = 0.1  # kWh/minute
discharge_rate = 0 # MODIFY:make it =0

charge_rate = discharge_rate * 2
infeasib_cost = 1e5
infeasib_threas = 1e4
travel_edge_time = 10
in_value = 1.0 / n_veh  # for initialization of x_ij

# Random SOC array
# min_charge = 6 # MODIFY: make it = 0 so EV never needs to charge.
min_charge = 0 # MODIFY: make it = 0 so EV never needs to charge.
intermediate_charge = 31
full_charge = 50  # kWh
random_soc = [[np.random.uniform(min_charge, full_charge) for n in range(n_veh)] for iter in range(num_iter)]


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
    # Choose between random and full_charge
    # MODIFY: make every vehicle soc = full_charge
    for j in range(n_veh):
        # vehicles[j].stateOfCharge = random_soc[it][j]
        vehicles[j].stateOfCharge = full_charge

    # Track charging
    y_power_cars = []
    cars_charging = 0

    # Variables to track performance
    miss_ride_time = []
    low_battery_time = []
    int_battery_time = []
    high_battery_time = []
    ev_ride_time = []
    ev_charge_time = []
    ev_idle_time = []

    h_format = []

    # Iterate over requests, deltaT = 1 min
    for k in range(tData.num_min):
        PULoc = tData.records[k][0]
        DOLoc = tData.records[k][1]
        minute = k + tData.h_in * 60
        numRideReq = len(PULoc)
        h_format.append(time.strftime("%H:%M", time.gmtime(minute * 60)))

        print("*** Minute: " + str(k) + " ***")

        req_vec = []  # List of requests collected between time t and t+1
        req_idx = 0

        start_costs = [0] * n_veh  # costs to get to the pick-up point
        start_path = [0] * n_veh  # path to get to the pick-up point

        cost = []
        paths = []

        # Update vehicle positions
        for j in range(n_veh):
            # If vehicle reaches passenger final destination
            if vehicles[j].request is not None and isinstance(vehicles[j].request, RideRequest) and vehicles[j].estimated_done_riding <= k:
                vehicles[j].position = vehicles[j].request.end
                vehicles[j].request = None  # Vehicle again available
            # If vehicle runs out of battery ---> must charge, becomes unavailable
            if vehicles[j].request is None and not vehicles[j].charging and vehicles[j].stateOfCharge < min_charge:
                vehicles[j].charging = True
                cars_charging += 1

        # Update vehicle state-of-charge
        for j in range(n_veh):
            if vehicles[j].estimated_done_riding >= k:
                vehicles[j].stateOfCharge -= discharge_rate
            elif vehicles[j].charging:
                vehicles[j].stateOfCharge += charge_rate
                # If vehicle is fully charged, disconnect
                if vehicles[j].stateOfCharge >= full_charge:
                    vehicles[j].stateOfCharge = full_charge
                    vehicles[j].charging = False
                    cars_charging -= 1

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
            elif v.charging:
                charge_cars += 1
        ev_ride_time.append(ride_cars)
        ev_charge_time.append(charge_cars)
        ev_idle_time.append(n_veh - (ride_cars + charge_cars))

        # Generate a ride request
        for i in range(numRideReq):
            ride_req = f.create_ride_request(PULoc[i], DOLoc[i], k, g, 0, 0)
            req_vec.insert(req_idx, ride_req)

            # Compute cost of reaching the pickup point for each vehicle
            for j in range(n_veh):
                current_pos = vehicles[j].position
                start_path[j] = nx.shortest_path(g, source=current_pos,
                                                 target=ride_req.start)  # Path from current vehicle node to start node
                if vehicles[j].request is not None or \
                        vehicles[j].stateOfCharge < min_consume + len(start_path[j]) + len(req_vec[req_idx].path) - 2 or \
                        vehicles[j].stateOfCharge < min_charge or \
                        vehicles[j].charging or \
                        delta_ride < len(start_path[j]) - 1:
                    start_costs[j] = infeasib_cost
                else:
                    start_costs[j] = len(start_path[j]) - 1

            cost.append(start_costs.copy())
            paths.append(start_path.copy())
            req_idx += 1

        if not req_idx:
            print("No real request at this time!")
            y_power_cars.append(cars_charging * power_transferred)
            miss_ride_time.append(0)
            continue
        elif sum([sum(cost[i]) for i in range(req_idx)]) == infeasib_cost * n_veh * req_idx:
            print("WARNING: No feasible request at this time!")
            y_power_cars.append(cars_charging * power_transferred)
            miss_ride_time.append(req_idx)
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

        # Assignment problem, only riding part
        Kout = 2  # Iterations

        xi = [[0 for col in range(Kout + 1)] for row in range(n_assign * n_assign)]
        # Initialize with xi inside the feasible set, i.e. satisfies the constraints.
        for i in range(n_assign * n_assign):
            xi[i][0] = in_value

        cost_function = np.zeros(n_assign * n_assign)

        for t in range(Kout):
            for jj in range(n_assign):  # Loop over requests - columns
                for ii in range(n_assign):  # Loop over vehicles - rows
                    req_j = ii * (1 + jj) + (n_assign - ii) * jj
                    cost_function[req_j] = cost[ii][jj]

            # Solve the outer problem using PuLP, a Python toolbox
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
            else:
                print("Wrong request assignment")

        y_power_cars.append(cars_charging * power_transferred)

        count_assigned_rides = 0
        for i in vehicles:
            if i.request in req_vec and isinstance(i.request, RideRequest):
                count_assigned_rides += 1
        if count_assigned_rides == req_idx:
            miss_ride_time.append(0)
        else:
            miss_ride_time.append(req_idx - count_assigned_rides)

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
    ev_ride_time_avg += np.array(ev_ride_time)
    ev_charge_time_avg += np.array(ev_charge_time)
    ev_idle_time_avg += np.array(ev_idle_time)

# Average data, no rounding yet
y_power_cars_avg = y_power_cars_avg/num_iter
miss_ride_time_avg = miss_ride_time_avg/num_iter
high_battery_time_avg = high_battery_time_avg/num_iter
int_battery_time_avg = int_battery_time_avg/num_iter
low_battery_time_avg = low_battery_time_avg/num_iter
ev_ride_time_avg = ev_ride_time_avg / num_iter
ev_charge_time_avg = ev_charge_time_avg / num_iter
ev_idle_time_avg = ev_idle_time_avg / num_iter

# Re-name variables
miss_ride_time = miss_ride_time_avg
y_power_cars = y_power_cars_avg
high_battery_time = high_battery_time_avg
int_battery_time = int_battery_time_avg
low_battery_time = low_battery_time_avg
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

print("QoS: " + str(100-(int(sum(np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1)))))/2462*100))

# Plots
# Pref and missing rides
plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 12})
plt.rcParams['axes.xmargin'] = 0

fig, host = plt.subplots(figsize=(4.2, 2.55), layout="constrained") # layout='constrained')  # (width, height) in inches
ax3 = host.twinx()
host.set_xlabel("Time")
host.set_ylabel("Power [kW]")
ax3.set_ylabel("Missed ride requests")
ax3.locator_params(axis="y", integer=True, tight=True)
p1b = host.plot(h_format, y_power_cars, label='$v_{\mathrm{ch}} p_{\mathrm{ch}}$', color="#1E5EE1")
p3 = ax3.bar(h_format[::time_slot], np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1)), width=7, alpha=0.3, color = "#C8377E", label=("Total missed ride requests: " + str(int(sum(np.rint(np.sum(np.array(miss_ride_time).reshape(len(miss_ride_time) // time_slot, time_slot), axis=1)))))))
ax3.legend(loc='upper center', fontsize="11", ncol=1, bbox_to_anchor=(0.48, 1.25))
host.legend(handles=p1b, loc='upper left', fontsize="12", ncol=1)
host.set_xticks(h_format[::360])
host.set_xticklabels(h_format[::360])
plt.savefig("NoChargeReq_MissRide_avg.pdf", bbox_inches='tight')

#SOC
plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 12})
plt.rcParams['axes.xmargin'] = 0

fig3, ax3 = plt.subplots(figsize=(4.3,3), tight_layout=True)
ax3.bar(h_format, high_battery_time, width=1.0, alpha=0.5, label="High SOC", color = "limegreen")
ax3.bar(h_format, int_battery_time, width=1.0, alpha=0.5, label="Mid SOC", color = "gold")
ax3.bar(h_format, low_battery_time, width=1.0, alpha=0.6, label="Low SOC", color = "orangered")
ax3.set_xticks(h_format[::360])
ax3.set_xticklabels(h_format[::360])
ax3.set_xlabel('Time')
ax3.set_ylabel('Number of EV')
ax3.margins(y=0)
ax3.legend(loc='upper center', fontsize="11",  bbox_to_anchor=(0.48, 1.25), ncol=3)
fig3.savefig('NoChargeReq_SOC_avg.pdf', bbox_inches='tight')

# EV request status
plt.rcParams.update({"text.usetex": True, "font.family": "lmodern", "font.size": 12})
plt.rcParams['axes.xmargin'] = 0

fig7, ax7 = plt.subplots(figsize=(4.2,3), tight_layout=True)
ax7.bar(h_format, ev_idle_time, width=1.0, alpha=0.6, label="Idling", color = "#C53AA6")
ax7.bar(h_format, ev_ride_time, width=1.0, alpha=0.5, label="Riding", color = "#3AA6C5")
ax7.bar(h_format, ev_charge_time, width=1.0, alpha=0.5, label="Charging", color = "#A6C53A")
ax7.set_xticks(h_format[::360])
ax7.set_xticklabels(h_format[::360])
ax7.set_xlabel('Time')
ax7.set_ylabel('Number of EV')
ax7.margins(y=0)
ax7.legend(loc='upper center', fontsize="11", bbox_to_anchor=(0.48, 1.25), ncol=3)
fig7.savefig('NoChargeReq_EV_avg.pdf', bbox_inches='tight')

plt.show()

# %%
