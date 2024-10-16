# MPC (Model Predictive Control) Steering using Sparse System Model Identification.

This is a MPC controller implmentation for the steering control challenge. The control model was identified using the `pysindy' python package and by doing some parameter optimization on it.

`pysindy' package provides tools for applying the sparse identification of nonlinear
dynamics (SINDy) of dynamical system using data traces taken from system runs. In our case I applied the identification to the rollout runs of the steering simulator to get an aproximated sparse symbolic model of the system. After, some parameter optimizations the model and controller performed well in the challenge ranking on the top 14 of the leader board (see details below).

The mpc controller code was adapted from a related problem (credit 1). 

Credits are due to:
  (1) The reference MPC implementation by Mark Misin (see also header in python code)
  (2) `pysindy' python package : (https://pypi.org/project/pysindy/).

## Usage
Follow the task's original instructions - use the `mpcMainParams` controller. Please note that the controller requires to install the python packages: `qpsolvers` and `cvxpy`. (see the requirements.txt file) 


## Methodology
I used the scripts in the `pysindy_optimization` directory to explore the model space for a symbolic model. 
<!--Also are scripts used to search model parameters that would optimize the controller.-->
In particular I used the sindy SR3 method for sparse identification. The model identification consisted of 
two basic steps:
* Collecting simulations runs (rollouts) traces of the system  
* Running the `pySindy` SR3 sparse identification with the rollout runs as inputs.

### Collecting simulations rollouts - 
- The simulation runs are done using the challenge's physical model simulater. The script `tinyphysics_opt.py` was adapted to include an extra option to store rollouts as csv files. Note that the
stored rollouts also include the control signals of any controller you might have selected. Actually. there is no importance to which controller, and I have found that mixing and batching rollouts from several controllers as input to the `sindy` algorithm provided the best model results.

An example of the command to collect rollout using the pid controller:

```
python tinyphysics_opt.py --model_path ../models/tinyphysics.onnx --data_path ../data --num_segs 10 --controller pid --collect
```
The rollouts are saved in a new folder: `rollout_result/`

### Running the `pySindy` script

Please note that pySindy package needs to installed to the script: steer_modelSR3_sindy.py
see instructions in `https://pypi.org/project/pysindy/`.
Also note that you would need to update the script to define the rollout sets you wish to use. (see the train_data, test_data variables in the code).

Operate the script with the following command:
```
python steer_modelSR3_sindy.py
```


## Comma Controls Challenge: Report Snippet

<body>
<h3 style="font-size: 30px; margin-top: 50px">Aggregate Costs (total rollouts: 5000)</h2>
<h3 style="font-size: 30px;"><span style="background: #c0392b; color: #fff; padding: 10px">Test Controller: mpcMainParams</span> ⚔️ <span style="background: #2980b9; color: #fff; padding: 10px">Baseline Controller: pid</span></h3>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>controller</th>
      <th>lataccel_cost</th>
      <th>jerk_cost</th>
      <th>total_cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>baseline</td>
      <td>2.351</td>
      <td>23.781</td>
      <td>141.308</td>
    </tr>
    <tr>
      <td>test</td>
      <td>1.422</td>
      <td>20.007</td>
      <td>91.130</td>
    </tr>
  </tbody>
</table>
<h3 style="font-size: 20px; color: #27ae60"> ✅ Test Controller (mpcMainParams) passed Baseline Controller (pid)! ✅ </h3>
</body>


# Comma Controls Challenge!
![Car](./imgs/car.jpg)

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.


## Getting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual car and road states from [openpilot](https://github.com/commaai/openpilot) users.

```
# install required packages
# recommended python==3.11
pip install -r requirements.txt

# test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller pid 
```

There are some other scripts to help you get aggregate metrics: 
```
# batch Metrics of a controller on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid

# generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller pid --baseline_controller zero

```
You can also use the notebook at [`experiment.ipynb`](https://github.com/commaai/controls_challenge/blob/master/experiment.ipynb) for exploration.

## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. Its inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`), and a steer input (`steer_action`), then it predicts the resultant lateral acceleration of the car.


## Controllers
Your controller should implement a new [controller](https://github.com/commaai/controls_challenge/tree/master/controllers). This controller can be passed as an arg to run in-loop in the simulator to autoregressively predict the car's response.


## Evaluation
Each rollout will result in 2 costs:
- `lataccel_cost`: $\dfrac{\Sigma(actual\\_lat\\_accel - target\\_lat\\_accel)^2}{steps} * 100$

- `jerk_cost`: $\dfrac{\Sigma((actual\\_lat\\_accel\_t - actual\\_lat\\_accel\_{t-1}) / \Delta t)^2}{steps - 1} * 100$

It is important to minimize both costs. `total_cost`: $(lataccel\\_cost * 50) + jerk\\_cost$

## Submission
Run the following command, then submit `report.html` and your code to [this form](https://forms.gle/US88Hg7UR6bBuW3BA).

```
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <insert your controller name> --baseline_controller pid
```

## Changelog
- With [this commit](https://github.com/commaai/controls_challenge/commit/fdafbc64868b70d6ec9c305ab5b52ec501ea4e4f) we made the simulator more robust to outlier actions and changed the cost landscape to incentivize more aggressive and interesting solutions.
- With [this commit](https://github.com/commaai/controls_challenge/commit/4282a06183c10d2f593fc891b6bc7a0859264e88) we fixed a bug that caused the simulator model to be initialized wrong.

## Work at comma

Like this sort of stuff? You might want to work at comma!
[comma.ai/jobs](https://comma.ai/jobs)
