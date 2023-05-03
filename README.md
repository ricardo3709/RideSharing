# Renewable-Based Charging in Green Ride-Sharing
Pyhton code that implements a Gauss-Seidel algorithm to promote renewable-based charging in a 100% -electrified fleet of vehicles for ride-hailing or ride-sharing services, as described in [1]. 

# Required Software
Python 3.8 or earlier: https://www.python.org/downloads/

cvxpy: https://www.cvxpy.org/install/

PuLP: https://pypi.org/project/PuLP/

# Data
Data for the ride requests needs to be downloaded from the Taxi and Limousine Commission https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
2022 >> March >> Yellow Taxi Trip Records (PARQUET)
The .parquet file should be saved in the same folder as the main files, listed below, under the name “yellow_tripdata_2022-03.parquet”

# Execution
Main files: To execute the cases described in [1], run as follow:

Business-as-usual case: run the file “businessAsUsualCase.py”, for the random SOC results, average over 10 experiments.
Case 1: run the file “case1.py”, for the sunny day results, average over 10 experiments.
Case 2: run the file “case2.py”, for the sunny day and 75% willingness to ride-share results, average over 10 experiments.
Fossil-fuel vehicles case: run the file “fossilFuelCase.py”.

The same folder that contains the main file above, should have:
·        tripData.py: code to prepare the ride request data from the “yellow_tripdata_2022-03.parquet” file.
·        ChargeRequest.py: charge request class.
·        RideRequest.py: ride request class.
·        Vehicle.py: vehicle class
·        functions.py: Miscellaneous functions
·        taxiZones.cvs: Taxi Zone Lookup Table
·        PV_norm.cvs: PV generation data for sunny, cloudy morning, and cloudy afternoon scenarios.

To repeat the experiments presented in [1], for conditions such as weather are initial SOC, follow:

Business-as-usual case:
In “Business-as-usual.py”:
·        Lines 82-83: select between “random SOC” or “fully charge”.
·        Line 27: should be 1 to run just one experiment.

Case 1:
In “case1.py”:
·        Lines 32-34: select between “PV_sunny”, “PV_cloud_am” or “PV_cloud_pm”.
·        Line 29: should be 10 to run the case for 10 experiments.

Case 2:
In “case2.py”:
·        Lines 35-37: select between “PV_sunny”, “PV_cloud_am” or “PV_cloud_pm”.
·        Line 29: should be 10 to run the case for 10 experiments.
·        Line 32: should be 0.75 for the case of 75% willingness to ride-share, 1 for 100%, 0.5 for 50%, and 0.25 for 25%.

 

# Documentation
[1] E. Perotti, A. M. Ospina, G. Bianchin, A. Simonetto, and E. Dall’Anese, “Towards the Decarbonization of the Mobility Sector: Promoting Renewable-Based Charging in Green Ride-Sharing,” arXiv preprint, May. 2023, (link).
