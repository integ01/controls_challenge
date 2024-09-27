'''
LICENSE AGREEMENT

    Written by: Ben Shimony

   This program is free software: you can redistribute it and/or modify it 
   under the terms of the GNU General Public License as published by the 
   Free Software Foundation, either version 3 of the License, or (at your option) 
   any later version.
    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
   without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
   See the GNU General Public License for more details.
   You should have received a copy of the GNU General Public License along with this program. 
   If not, see <https://www.gnu.org/licenses/>. 
'''

import numpy as np
import matplotlib.pyplot as plt
from . import BaseController
from . import carSteer_state_spaceParams as sfc_g
from qpsolvers import *
from collections import namedtuple
import pdb

Ts = 0.
outputs = 0
inputs = 0
MAX_SEQLEN= 600
MAX_HORIZON = 50
VER =5
State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])
support= None#sfc_g.SupportFilesCar()
duOk_g = True

def printD(msg):
    pass#print(msg)

def calc_long_u_accel(v_ego, a_ego):
    area = 0.74 #m**2
    drag_coef =  0.35
    air_density = 1.225 #kg/m**3
    vel_square = v_ego**2 #m/sec
    car_weight = 1400 #kg
    car_mu=0.02 # friction coefficient
    a_drag = 0.5*air_density*drag_coef*area*vel_square/car_weight
    a_fric = car_mu * 9.81 
    long_accel = a_drag + a_ego + a_fric# - df['actual_lataccel']

    return long_accel

'''
args = { 'hz': 10, 'r11': 350, 'r22':100,  

       a33_ydd_0 = 0.989
        a33_ydd_1xv = -1e-3
        b31_u0_1xv = 0.002
        b33_ang_0 = 0.003
        b33_ang_1xv = 0.001
        b33_ang_2ydd = 0.002
}
''' 
class Controller(BaseController):

  def set_params(self, kwargs):
    if self.version == 1:
        r11 = kwargs['r11']
        r22 = kwargs['r22']
        self.support.constants['R']=np.matrix([[r11, 0, 0],[0, r22, 0], [0, 0, 1]])
        #self.support.constants['hz'] = kwargs['hz']
        self.support.constants['a33_ydd_0'] = kwargs['a33_ydd_0']
        self.support.constants[ 'a33_ydd_1xv']= kwargs['a33_ydd_1xv']
        self.support.constants[ 'b31_u0_1xv'] = kwargs['b31_u0_1xv']
        self.support.constants[ 'b33_ang_0'] = kwargs['b33_ang_0']
        self.support.constants['b33_ang_1xv'] = kwargs['b33_ang_1xv']
        self.support.constants[ 'b33_ang_2ydd']= kwargs['b33_ang_2ydd']
    elif self.version == 5:
        if 'r11' in kwargs:
            R11 = kwargs['r11']
            self.support.constants['R']=np.matrix([[R11, 0, 0],[0, R11/2, 0], [0, 0, 1]])
        if 'q11' in kwargs:
            Q11 = kwargs['q11']
            self.constants['Q']=np.matrix([[Q11, 0, 0],[0, Q11/2, 0], [0, 0, Q11*3]]) 
            S11 = Q11*0.9
            self.constants['S']=np.matrix([[S11, 0, 0],[0, S11/2, 0], [0, 0, S11*3]])
        if 's11' in kwargs:
            S11 = kwargs['s11']
            self.constants['S']=np.matrix([[S11, 0, 0],[0, S11/2, 0], [0, 0, S11*3]])
        if 'pid_alpha' in kwargs:
            self.pid_alpha = kwargs['pid_alpha']
        else:
            self.pid_alpha = 0.001

        if 'a33_ydd_0' in kwargs:
            self.support.constants['a33_ydd_0'] = kwargs['a33_ydd_0']
        if 'a33_ydd_1xv' in kwargs:
            self.support.constants[ 'a33_ydd_1xv']= kwargs['a33_ydd_1xv']
        if 'b31_u0_0' in kwargs:
            self.support.constants[ 'b31_u0_0'] = kwargs['b31_u0_0']
        if 'b31_u0_1xdv' in kwargs:
            self.support.constants[ 'b31_u0_1xdv'] = kwargs['b31_u0_1xdv']
        if 'b31_u0_2xydd' in kwargs:
            self.support.constants[ 'b31_u0_2xydd'] = kwargs['b31_u0_2xydd']
        if 'b31_u0_3xa' in kwargs:
            self.support.constants[ 'b31_u0_3xa'] = kwargs['b31_u0_3xa']
        if 'b31_u0_4xang' in kwargs:
            self.support.constants[ 'b31_u0_4xang'] = kwargs['b31_u0_4xang']
        if 'b12_u1_0' in kwargs:
            self.support.constants['b12_u1_0'] = kwargs['b12_u1_0'] 
        if 'a33_ydd_2xydd' in kwargs:
            self.support.constants['a33_ydd_2xydd'] = kwargs['a33_ydd_2xydd']
        if 'b33_ang_0' in kwargs:
            self.support.constants[ 'b33_ang_0'] = kwargs['b33_ang_0']
        if 'b33_ang_1xv' in kwargs:
            self.support.constants['b33_ang_1xv'] = kwargs['b33_ang_1xv']
    

  def __init__(self,version=VER):
    global output, inputs, support
    global duOk_g
    # Create an object for the support functions.

    self.version = version
    self.support= sfc_g.SupportFilesCar(version = version)#**kwargs)
    self.constants=self.support.constants
    self.outputs=self.constants['outputs'] # number of outputs
    self.inputs=self.constants['inputs'] # number of inputs controls (steer, agas, roll)
    self.hz = self.constants['hz'] # horizon inputs length 
    #printD(f"self.hz:{self.hz}")
    #printD(f"self.version:{self.version}")
    support = self.support
    if version == 1:
        support.constants['Q']=np.matrix([[600, 0, 0],[0, 300, 0], [0, 0, 1640]]) 
        support.constants['S']=np.matrix([[300, 0, 0],[0, 150, 0], [0, 0, 800]]) 

        kwargs = { 'r11': 7000, 'r22': 3960, 'a33_ydd_0': 0.99, 'a33_ydd_1xv': -0.0011, 'b31_u0_1xv': 0.0011, 'b33_ang_0': 0.001, 'b33_ang_1xv': 0.0003, 'b33_ang_2ydd': 0.001}
    else:
        support.constants['Q']=np.matrix([[600, 0, 0],[0, 300, 0], [0, 0, 1600]]) 
        support.constants['S']=np.matrix([[300, 0, 0],[0, 150, 0], [0, 0, 800]]) 
    
        # Default org with Added params:
        #kwargs =  {'r11': 3200, 'q11': 600, 's11': 400, 'a33_ydd_0': 0.943,  'a33_ydd_1xv':-0.001,
        #'b31_u0_0': 0.204, 'b31_u0_1xdv': -0.002, 'b31_u0_2xydd':0.01,'b31_u0_3xa':0.017,
        #'b33_ang_0': 0.079 ,'b31_u0_4xang': 0.000, 'b12_u1_0':0.092, 'a33_ydd_2xydd': -6e-3,
        #'b33_ang_1xv' :0. }

        #Best optimize
        kwargs = { 'a33_ydd_0': 0.782, 'a33_ydd_1xv': 0.004, 'a33_ydd_2xydd': -0.004375047076229112, 'b31_u0_0': 0.32, 'b31_u0_1xdv': -0.002, 'b31_u0_2xydd': 0.02, 'b31_u0_3xa': 0.02751268189269432, 'b31_u0_4xang': -0.003, 'b12_u1_0': 0.2, 'b33_ang_0': 0.05240590097851129, 'b33_ang_1xv':-0.6e-3}
        
        kwargs['r11']=1700#2000#3900#6609
        kwargs['q11']=30#100#130#100
        kwargs['s11']=27#90#120#60
        kwargs['pid_alpha']=0.036#0.039#0.017
    
    printD(self.support.constants)
    outputs = self.outputs
    inputs = self.inputs
    self.refSignals=np.zeros(MAX_HORIZON*outputs)
    self.refU = np.zeros(MAX_HORIZON*inputs)
    self.steer_history = []
    self.step_idx =0
    self.solverNonErrorCount = 0
    self.solverErrorCount = 0

    self.du=np.zeros((inputs*self.hz,1))
    # Set global constant values.
    Ts=self.constants['Ts']

    # Build up the reference signal vector:
    '''
        k=0 
        for i in range(0,len(refSignals),outputs):
            refSignals[i]=x_train[k,0]
            refSignals[i+1]=x_train[k,1]
            refSignals[i+2]=x_train[k,2]
            k=k+1
    '''
    self.t = np.linspace(0, (MAX_SEQLEN+1)*Ts,MAX_SEQLEN)
    self.states = None 
    #  keep track of all your states during the entire manoeuvre
    self.statesTotal=np.zeros((len(self.t),self.outputs))
    self.ucontrolTotal= np.zeros((len(self.t),self.inputs))
    self.du = np.zeros((inputs*self.hz,1))
    self.u1 = 0
    ######### Back up Pid controller - used with the Integrator only 
    self.backupPid = True
    self.i = 0.05
    self.error_integral = 0
    self.prev_error = 0
    #self.pid_alpha = 0.335# 0.01
    ####

    # Set all parameters
    self.set_params(kwargs)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    #current_states = [target_lataccel, lataccel, state.rollaccel, state.v_ego, state.a_ego]
    ## Target step_idx+1 :step+FUTURE_STEPS
    lataccelRefs =future_plan.lataccel
    rollaccelRefs =future_plan.roll_lataccel
    v_egoRefs =future_plan.v_ego
    a_egoRefs =future_plan.a_ego
    
    # Build up the reference signal vector:
    refSignals = zip(v_egoRefs, a_egoRefs, lataccelRefs)

    u2 = [ calc_long_u_accel(v_ego, a_ego) for v_ego, a_ego in zip(v_egoRefs,a_egoRefs)]
    u2.insert(0,calc_long_u_accel(state.v_ego, state.a_ego)) # insert the first current state at the start
    u3 = [state.roll_lataccel]+ rollaccelRefs
    if self.step_idx == 0:
        self.ucontrolTotal[0,:] = np.array([self.u1,u2[0],u3[0]])
    u1 = np.zeros(len(future_plan.lataccel)+1)
    printD(f"================= Step {self.step_idx} ====================")
    printD(f"future:{len(future_plan.lataccel)},hz :{self.hz}")
    if u1.shape[0] < self.hz:
        self.hz = u1.shape[0]
        printD(f"---->>>> Chaged hz :{self.hz}")
    printD(f"stepidx:{self.step_idx}, hz :{self.hz}")
    if self.step_idx == 80:    ## Add random noise to first control actuation:
      	self.ucontrolTotal[self.step_idx:self.step_idx+self.hz,0] += np.random.normal(0,0.15,self.hz)
    if self.step_idx >= 80:   ### Update planned control from past estimations
        #    u1 = [target_lataccel] + lataccelRefs
        u1[:self.hz] = self.ucontrolTotal[self.step_idx:self.step_idx+self.hz,0] 
        u2[:self.hz] = self.ucontrolTotal[self.step_idx:self.step_idx+self.hz,1] 

    # Build up the control vector:
    refControls =  zip(u1,u2,u3)
    # Set the current state in log:
    self.statesTotal[self.step_idx,2] =current_lataccel
    self.statesTotal[self.step_idx,1] =state.a_ego
    self.statesTotal[self.step_idx,0] =state.v_ego

    ## Back up pid integrator
    error = (target_lataccel - current_lataccel)
    #error_future_avg = np.mean(np.array([target_lataccel]+lataccelRefs)[:self.hz+1])-current_lataccel

    self.error_integral += error
    steerPid = self.error_integral 
    
    #### Start of Calculations ####
    steerCmd = 0. 
    
        ######  Get Mpc estimation (only after time-step over 90)  ######
    if self.step_idx > 81-self.hz: 
        steerCmd = self.estimateMpcStep( refSignals, refControls)
        if steerCmd is not None:
            printD(f"new steer:{steerCmd}")

            if steerCmd > 2.0: steerCmd=2.0
            if steerCmd < -2.0: steerCmd=-2.0
        else:
            printD("Got None from optimizer - using last value")
            steerCmd = u1[0]# np.max(1-error,0.1)* u1[0]+ error * steerBackPid
            printD(f"new steer:{steerCmd}")
            printD(f"lataccel: {current_lataccel}")
    else:
        self.ucontrolTotal[self.step_idx,0] = 0

        #### Combine MPC with Pid integrator ####
    if self.backupPid:
        steerCmd = (1.-self.pid_alpha)*steerCmd+ self.pid_alpha*steerPid
    self.step_idx += 1
    return steerCmd     

  #################################################################################
  # estimateMpcStep(self, refSignals_, refControls_):
  # In: 
  #     refSignals: tuple list of current and future system trajectory 
  #                 zip(v_egoRefs, a_egoRefs, lataccelRefs)
  #     refControls: tuple list of future actuations as calulated and accumelated from environmet
  #                  updates:  zip(u1,u2,u3)
  #                   u1 - esrimated wheel-steer acceleration control.
  #                   u2 - estimated car pedal gas acceleration.
  #                   u3 - [state.roll_lataccel]+ rollaccelRefs
  #     length of tuple lists is the MPC planning horizon = self.hz.
  #     Function is an adaption based on script by Mark Misin. See full copyright in the 
  #     carSteer_state_spaceParams.py file.
  # Out:
  #      Steer cmd - wheel steer force acceleration 
  ##################################################################################################
  def estimateMpcStep(self, refSignals_:list, refControls_:list)-> float:

    global duOk_g
    refSignals = list(refSignals_)
    refControls = list(refControls_)
    assert (len(refSignals)+1 == len(refControls)),"My assert error: estimate inputs lengths not the same"
    
    next_hz = min(self.hz, len(refSignals))
    #for row,i in enumerate(range(0,len(refSignals),self.outputs)):
    printD("--- States - Vego, Accel ego, latter Accel : -----")
    for row,i in enumerate(range(0,next_hz*self.outputs,self.outputs)):
        printD(refSignals[row])
        self.refSignals[i]= refSignals[row][0] #v_ego
        self.refSignals[i+1]= refSignals[row][1] #a_ego
        self.refSignals[i+2]= refSignals[row][2] #latAccel

    printD("--- Ref Contorl in : Steer, GadPedal, Road Roll Accel-----")
    for row, i in enumerate(range(0,(next_hz+1)*self.inputs,self.inputs)):
        if row == 0:
            printD(f"{refControls[row]} <<<<<<<<")
        else:
            printD(refControls[row])
        self.refU[i] =  refControls[row][0] #target steer
        self.refU[i+1] =  refControls[row][1] #a gas
        self.refU[i+2] =  refControls[row][2] # roll_accel


    u1 = refControls[0][0]
    u2=  refControls[0][1]
    u3 = refControls[0][2]
   

    states = self.statesTotal[self.step_idx,:]

    printD (f"States {self.step_idx}: {states}")
      # Generate the discrete state space matrices
    Ad,Bd,Cd,Dd=self.support.state_space(states,u1,u2,u3)
    #print(f"Ad:{Ad}\nBd:{Bd},\nCd:{Cd}")
    # Generate the augmented current state and the reference vector
    x_aug_t=np.transpose([np.concatenate((states,np.array([u1,u2,u3])),axis=0)])
    # From the refSignals vector, only extract the reference values 
    # from your [current sample (NOW) + Ts] to [NOW+horizon period (hz)]
    k = (self.step_idx+1)*self.outputs
    r=self.refSignals[:self.hz*self.outputs]

    # Generate the compact simplification matrices for the cost function
    Hdb,Fdbt,Cdb,Adc,G,ht=self.support.mpc_simplification(Ad,Bd,Cd,Dd,self.hz,x_aug_t,self.refU[:inputs*self.hz,np.newaxis])
    ft=np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)],r),axis=0),Fdbt)
    ######################### Constraints #####################################
    duOk = duOk_g
    try:  
        if duOk:
            self.du=solve_qp(Hdb,ft,G,ht,solver="cvxopt")   
            self.du=np.transpose([self.du])#np.random.random(0.5,1)
            self.solverNonErrorCount += 1
            #print(self.du)
            # exit()
            pass
    except Exception:  #ValueError as ve:
        printD(i,self.du)
        self.solverErrorCount += 1
        printD(f"Solver Error:{self.solverErrorCount}\r")
        duOk = False
    #############################################################################

    # # No constraints
    if not duOk or self.du[0] is None:
        self.du=-np.matmul(np.linalg.inv(Hdb),np.transpose([ft]))
        #print("NONE")#self.du)
        #pdb.set_trace() 
        duOk = True

    printD(self.du)
    # Update the real inputs
    try:
        if self.du is not None:
            #         Add  u1 + du (and clip) 
            u1=u1+self.du[0][0]
            u1=np.clip(u1,-2.,2.)#-0.5,0.5)
            #         Add  u2 + du  
            u2=u2+self.du[0][1]
        else:
            duOk = False
            print("du is None")
    except Exception:
        duOk = False
        pass
     
    if len(refControls) > 1:
        u3 =  refControls[1][2]
    else:
        pass
        #print(f"refControls:{refControls}, step:{self.step_idx}")
    

    # Setting next steer command:
    # self.ucontrolTotal[self.step_idx+1,:] = np.array([u1,u2,u3])
    self.ucontrolTotal[self.step_idx+1][2]=u3
    printD("--- next du's in : -----")
    printD (f"self.ucontrolTotal[{self.step_idx+1}](u1,2,3) :{u1} {u2} {u3} <<<<<<< ")
    for ei,i in enumerate(range(min(MAX_SEQLEN,self.step_idx+self.hz), self.step_idx,-1 )):
     #We assume self.step_idx > 80:
        self.ucontrolTotal[i,0] = self.ucontrolTotal[i-1,0] + np.clip(self.du[(i-self.step_idx-1)*inputs][0],-2.0,2.)
        self.ucontrolTotal[i,1] = self.ucontrolTotal[i-1,1] + np.clip(self.du[(i-self.step_idx-1)*inputs+1][0],-5.0,5.)
        if ei == self.hz-1:
            printD (f"self.ucontrolTotal[{i}] :{self.ucontrolTotal[i,:]} <<<<<<<")#, {self.du[(ei+1)*inputs][0]}")
        else:
            printD (f"self.ucontrolTotal[{i}] :{self.ucontrolTotal[i,:]}")#, {self.du[(ei+1)*inputs][0]}")
    #assert u1 == self.ucontrolTotal[self.step_idx+1,0],f"u1 != ucontrol0 at {self.step_idx}"
    if self.ucontrolTotal[self.step_idx+1,0]!= u1 and self.step_idx > 81:
        printD(f"u1 != ucontrol0 at {self.step_idx} u1:{u1}, ucontrol:{self.ucontrolTotal[self.step_idx+1,0]}")
    return u1
