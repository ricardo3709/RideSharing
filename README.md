# Renewable-Based Charging in Green Ride-Sharing

Python code that implements a Gauss-Seidel algorithm to promote renewable-based charging in a 100% -electrified fleet of vehicles for ride-hailing or ride-sharing services, as described in [1]. 

# Required Software

Python 3.8 or earlier: https://www.python.org/downloads/

cvxpy: https://www.cvxpy.org/install/

PuLP: https://pypi.org/project/PuLP/

# Data

Data for the ride requests can be downloaded from the Manhattan Taxi and Limousine Commission website https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page, selecting 2022 >> March >> Yellow Taxi Trip Records (PARQUET).

The .parquet file must be saved in the same folder as the main files, listed below, under the name <em>yellow_tripdata_2022-03.parquet</em>.

# Execution

To reproduce the cases described in [1], run the main files below as follow:

Business-as-usual case: run the file <em>businessAsUsualCase.py</em>, for the random SOC results, averaged over 10 experiments.

Case 1: run the file <em>case1.py</em>, for the sunny day results, average over 10 experiments.

Case 2: run the file <em>case2.py</em>, for the sunny day and 75% willingness to ride-share results, averaged over 10 experiments.

Fossil-fuel vehicles case: run the file <em>fossilFuelCase.py</em>.

The same folder that contains the main files above, should also have:
<ul>
  <li>tripData.py: code to prepare the ride request data from the <em>yellow_tripdata_2022-03.parquet</em> file</li>
  <li>ChargeRequest.py: charge request class</li>
  <li>RideRequest.py: ride request class</li>
  <li>Vehicle.py: vehicle class</li>
  <li>functions.py: various functions</li>
  <li>taxiZones.cvs: taxi zone lookup table</li>
  <li>PV_norm.cvs: PV generation data for sunny, cloudy morning, and cloudy afternoon scenarios</li>
</ul>

To repeat the experiments presented in [1], for different weather conditions or initial SOC:

In <em>Business-as-usual.py</em>:

·        Lines 82-83: uncomment “random SOC” or “fully charge” to select the initial SOC conditions

·        Line 27: set to 1 in order to run just one experiment

In <em>case1.py</em>:

·        Lines 32-34: uncomment “PV_sunny”, “PV_cloud_am” or “PV_cloud_pm” to select the weather scenario

·        Line 29: set to 10 in order to run the experiment 10 times 

In <em>case2.py</em>:

·        Lines 35-37: uncomment “PV_sunny”, “PV_cloud_am” or “PV_cloud_pm” to select the weather scenario

·        Line 29: set to 10 in order to run the experiment 10 times

·        Line 32: set to 0.75 for the case where the willingness to ride-share is 75%, or 1 for 100%, 0.5 for 50%, and 0.25 for 25%

# Documentation

[1] E. Perotti, A. M. Ospina, G. Bianchin, A. Simonetto, and E. Dall’Anese, “Towards the Decarbonization of the Mobility Sector: Promoting Renewable-Based Charging in Green Ride-Sharing”. 
