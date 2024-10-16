'''
  Adapted by: Ben Shimony

  The SR3 method for sparse identification is applied to steering challenge. 
  see: https://github.com/commaai/controls_challenge
  The result model is used for a MPC controller, see my code: 'controllers/mpcMainParams.py'

  Credits:

   - `pysindy' python package : https://pypi.org/project/pysindy/  and
                                https://pysindy.readthedocs.io/en/latest/
   - Code is based on examples from : https://github.com/dynamicslab/pysindy
     In particular jupyter code 'Summary of PySINDy YouTube tutorial videos':
     https://github.com/dynamicslab/pysindy/blob/master/examples/15_pysindy_lectures.ipynb
   - Paper: 'Sparse identification ofnonlinear dynamics for modelpredictive control in thelow-data limit'
             E. Kaiser etal.

  

##############################################################################
#  Revision notes:
#  v1 - First verision used model with 4 states- 
#        'v_ego' (xdot)- velocity, 
#        'a_ego' (xdotdot) - longtitude acceleration
#        'lataccel' (ydotdot) - lateral acceleration
#        ydot - aproximated lateral speed  
#      3 controls -
#        'steer_command' - actual steering force
#        'gas_brake_estimate' - aproximated longtitude controls.
#        'roll_lataccel' - sinusodial of road tilt angle.
#  v2 - Use 3 states and 3 control - removed the ydot state compared to v1,
#       Based on my local code: steer_modelSR3_sindy2v2.py
#
##############################################################################
'''
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pysindy as ps
from pysindy.differentiation import FiniteDifference
from steer_getdata import get_sim_run_data2, get_sim_run_data
import pdb
from sklearn.metrics import mean_squared_error



G = 9.81
dt = 0.1
STARTIDX=110
LEN = 250
DISCRETE=True
VER=5
NUM_OF_RUNS=70
PATH_ROLLOUT = './rollout_result'

'''
## Data fields -     roll_lataccel      v_ego     a_ego  target_lataccel  steer_command  actual_lataccel
'''


def get_data_paths(prefix, st, end):
    data_paths = [prefix+f"{i}.csv" for i in range(st,end)]
    return data_paths

# Initialize custom SINDy library so that we can have x_dot inside it.
library_functions = [
    lambda x: x,
    lambda x: 1.0/x,
    lambda x, y: x * y,
    lambda x: x ** 2,
    lambda x, y, z: x * y * z,
    lambda x, y: x * y ** 2,
    lambda x: x ** 3,
    lambda x, y, z, w: x * y * z * w,
    lambda x, y, z: x * y * z ** 2,
    #lambda x, y: x * y ** 3,
    #lambda x: x ** 4,
]
x_dot_library_functions = [lambda x: x]

# library function names includes both
# the x_library_functions and x_dot_library_functions names
library_function_names = [
    lambda x: x,
    lambda x: "1/"+x,
    lambda x, y: x + y,
    lambda x: x + x,
    lambda x, y, z: x + y + z,
    lambda x, y: x + y + y,
    lambda x: x + x + x,
    lambda x, y, z, w: x + y + z + w,
    lambda x, y, z: x + y + z + z,
    #lambda x, y: x + y + y + y,
    #lambda x: x + x + x + x,
    lambda x: x,
]


#t = np.linspace(0, 60,int(60/dt))
#t = np.linspace(STARTIDX*dt, (STARTIDX+LEN+1)*dt,LEN)
t = np.arange(LEN)*dt
# Control input

####################################
# Load simulation data runs
data_paths = get_data_paths(PATH_ROLLOUT +'/pid_run',10,10+ NUM_OF_RUNS)
data_paths += get_data_paths(PATH_ROLLOUT + '/zero_run',0, NUM_OF_RUNS)
data_paths += get_data_paths(PATH_ROLLOUT+ '/mpcMainParams_run',20, 20+NUM_OF_RUNS)
train_data = data_paths[:NUM_OF_RUNS-5]+data_paths[NUM_OF_RUNS+5:NUM_OF_RUNS*3-10]
#train_data = data_paths[:NUM_OF_RUNS-20]
test_data = data_paths[NUM_OF_RUNS-5:NUM_OF_RUNS+5]+data_paths[NUM_OF_RUNS*3-10:]
#test_data = data_paths[NUM_OF_RUNS-20:]
if VER==1:
    x_train, u_control, data_names = get_sim_run_data(train_data,st_idx= STARTIDX,seq_len=LEN)
    x_test, u_test, data_names = get_sim_run_data(test_data,STARTIDX, LEN,0.1)
else:
    x_train, u_control, data_names = get_sim_run_data2(train_data,st_idx= STARTIDX,seq_len=LEN)
    x_test, u_test, data_names = get_sim_run_data2(test_data,STARTIDX, LEN,0.1)

t_train_multi = [ t for _ in range(len(x_train))]

###################################
# Define pySindy models
poly_library = ps.PolynomialLibrary()
poly_library.fit([ps.AxesArray(x_train, {"ax_sample": 0, "ax_coord": 1})])
n_features = poly_library.n_output_features_
print(f"Features ({n_features}):", poly_library.get_feature_names())
#Features (15): ['1', 'x0', 'x1', 'x2', 'x3', 'x0^2', 'x0 x1', 'x0 x2', 'x0 x3', 'x1^2', 'x1 x2', 'x1 x3', 'x2^2', 'x2 x3', 'x3^2']
#

library_functions = [lambda x: 1.0 / (x )]#, lambda x: np.exp(-x)]
library_function_names = [
    lambda x: "1.0 / (" + x + ")",
   # lambda x: "exp(-" + x + ")",
]
custom_library = ps.CustomLibrary(
    library_functions=library_functions, 
    function_names=library_function_names
)

n_targets = x_train[0].shape[1]
constraint_rhs = np.array([5, 0])

# One row per constraint, one column per coefficient
if VER==1:
    outidx = 3
    constraint_lhs = np.zeros((2,72))# (n_targets *n_features))
else:
    outidx=2
#    constraint_lhs = np.zeros((2,54 ))# (n_targets *n_features))
    constraint_lhs = np.zeros((2,42 ))# (n_targets *n_features))

constraint_lhs[0, outidx] = 1 #x3 < 5 ydodot
constraint_lhs[1, 0] = 1 #x1 > 0
#constraint_lhs[1, outidx] = 1 #x3 > -5
#constraint_lhs[2, 0] = 1 #x3 > -5


#sr3_optimizer = ps.SR3(threshold=0.1, thresholder="l0")
sr3_optimizer = ps.ConstrainedSR3(
        constraint_rhs= constraint_rhs, 
        constraint_lhs= constraint_lhs.reshape(1,-1),
        inequality_constraints=True,
        thresholder="l1",
        tol=1e-7,
        threshold=10,
        max_iter=10000,
        )

model = ps.SINDy(optimizer=sr3_optimizer,
        feature_names=data_names,
        feature_library=poly_library,
        differentiation_method=ps.FiniteDifference(drop_endpoints=True),
        discrete_time=DISCRETE
        )
model.fit(x_train, u=u_control, t=t_train_multi, multiple_trajectories=True)

print("#### SR3 Model: ####\n")
model.print()
print("\n####################")
print(poly_library.get_feature_names())

print ("rmse on test data simulation:")
test_error_sum = 0.
for i in range (len(u_test)):
 try:
    if DISCRETE:
  
        x_test_sim = model.simulate(x_test[i][0],u= u_test[i],t=len(u_test[0]))
        test_error_sum = mean_squared_error(x_test_sim[:,outidx], x_test[i][:,outidx])**0.5
    else: 
        x_test_sim = model.simulate(x_test[i][0],t,u= u_test[i])
        test_error_sum = mean_squared_error(x_test_sim[:,outidx], x_test[i][:,outidx])**0.5
    print("test: {}, Model score: {}, mse_error: {}".format(i, model.score(x_test[i], u=u_test[i], t=dt), test_error_sum))
 except Exception as e: 
     print("simulate error :{e}")

input("press any key to exit")
