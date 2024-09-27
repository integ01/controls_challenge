'''
LICENSE AGREEMENT

In relation to this Python file:

1. Copyright of this Python file is owned by the author: Mark Misin
2. This Python code can be freely used and distributed
3. The copyright label in this Python file such as

copyright=ax_main.text(x,y,'© Mark Misin Engineering',size=z)
that indicate that the Copyright is owned by Mark Misin MUST NOT be removed.

WARRANTY DISCLAIMER!

This Python file comes with absolutely NO WARRANTY! In no event can the author
of this Python file be held responsible for whatever happens in relation to this Python file.
For example, if there is a bug in the code and because of that a project, invention,
or anything else it was used for fails - the author is NOT RESPONSIBLE!
'''
'''
Latest modifications by Ben Shimony:

This file was modified for wheel steering application challenge by : Ben Shimony
This Python code can be freely used and distributed.
The copyright label and latest modifications notes in this Python file should not be removed.
'''



import numpy as np
import matplotlib.pyplot as plt
import pdb

def calc_long_u_accel(df):
    area = 0.74 #m**2
    drag_coef =  0.35
    air_density = 1.225 #kg/m**3
    vel_square = df['v_ego']**2 #m/sec
    car_weight = 1400 #kg
    car_mu=0.02 # friction coefficient
    a_drag = 0.5*air_density*drag_coef*area*vel_square.values/car_weight
    a_fric = car_mu * 9.81 
    long_accel = a_drag + df['a_ego'].values + a_fric# - df['actual_lataccel']

    return long_accel

class SupportFilesCar:
    ''' The following functions interact with the main file'''

    def __init__(self,version):
        ''' Load the constants that do not change'''

        # Constants
        g=9.81
        m=1500
        Iz=3000
        Cf=38000
        Cr=66000
        lf=2
        lr=3
        Ts=0.1 #0.02
        mju=0.02 # friction coefficient

        ####################### Lateral control #################################
        FPS=10
        self.version = version

        outputs=3 # number of outputs
        inputs=3 # number of inputs
        #        TODO
        hz = 9#1*FPS # horizon period (seconds or samples????)

        #trajectory=3 # Choose 1, 2 or 3, nothing else
        #version=2 # This is only for trajectory 3 (Choose 1 or 2)

        # Matrix weights for the cost function (They must be diagonal)

        #if trajectory==3 and version==2:
            # Weights for trajectory 3, version 2
        Q=np.matrix('100 0 0 ;0 1000 0 ;0 0 10000') # weights for outputs (all samples, except the last one)
        S=np.matrix('50 0 0;0 500 0;0 0 5000') # weights for the final horizon period outputs
        R=np.matrix('10 0 0;0 1 0; 0 0 1') # weights for inputs
                # Please do not modify the time_length!
        delay=0

        ################ Model parameters #########################
        if self.version == 1: ## Version 1
            r11 = 350
            r22 = 100
        
            a33_ydd_0 = 0.989
            a33_ydd_1xv = -1e-3
            b31_u0_1xv = 0.002
            b33_ang_0 = 0.003
            b33_ang_1xv = 0.001
            b33_ang_2ydd = 0.002

            self.constants={'g':g,'m':m,'Iz':Iz,'Cf':Cf,'Cr':Cr,'lf':lf,'lr':lr,\
            'Ts':Ts,'mju':mju,'Q':Q,'S':S,'R':R,'outputs':outputs,'inputs':inputs,\
            'hz':hz,\
            'a33_ydd_0' : a33_ydd_0, 'a33_ydd_1xv':  a33_ydd_1xv,\
            'b31_u0_1xv' : b31_u0_1xv, 'b33_ang_0' : b33_ang_0,\
            'b33_ang_1xv' : b33_ang_1xv, 'b33_ang_2ydd' : b33_ang_2ydd}
            self.constants['R']=np.matrix([[r11, 0, 0],[0, r22, 0], [0, 0, 1]]) 
        else: #Version 5 (100 runs)
            r11 = 350
            q11 = 600
            s11 = 300
        
            a33_ydd_0 = 0.989
            a33_ydd_1xv = -1e-3
            b31_u0 = 0.164
            b31_u0_1xv = 0.002
            b33_ang_0 = 0.079
            b33_ang_1xv = 0.001
            b33_ang_2ydd = 0.002
            b12_u1 = 0.097              
            b31_u0_2 = 0.032
            self.constants={'g':g,'m':m,'Iz':Iz,'Cf':Cf,'Cr':Cr,'lf':lf,'lr':lr,\
            'Ts':Ts,'mju':mju,'Q':Q,'S':S,'R':R,'outputs':outputs,'inputs':inputs,\
            'hz':hz,\
            'a33_ydd_0' : a33_ydd_0, 'a33_ydd_1xv':  a33_ydd_1xv,\
            'b31_u0_0' : b31_u0, 'b31_u0_1xv' : b31_u0_1xv, 'b33_ang_0' : b33_ang_0,\
            'b33_ang_1xv' : b33_ang_1xv, 'b33_ang_2ydd' : b33_ang_2ydd}
            self.constants['R']=np.matrix([[r11, 0, 0],[0, r11/2, 0], [0, 0, 1]]) 
            self.constants['Q']=np.matrix([[q11, 0, 0],[0, q11/2, 0], [0, 0, q11*3]]) 
            self.constants['S']=np.matrix([[s11, 0, 0],[0, s11, 0], [0, 0, 1]]) 
            self.constants['b12_u1_0'] = b12_u1
            self.constants['b31_u0_2xang'] = b31_u0_2
            self.constants['b33_ang_0'] = 0.079
            self.constants['b33_ang_1xv'] = 0.

        return None
    
 
    def state_space(self,states,delta,a, ang):
        '''This function forms the state space matrices and transforms them in the discrete form'''
        # Get the necessary constants
        g=self.constants['g']
        m=self.constants['m']
        Iz=self.constants['Iz']
        Ts=self.constants['Ts']
        mju=self.constants['mju']

        # Get the necessary states
        x_dot=states[0]
        xddot=states[1]
        yddot=states[2]
        #ang=states[3]
        '''
    xdot)[k+1] = 0.703 xdot[k] + 0.009 xdot[k]^2 + 0.006 xdot[k] yddot[k] + 0.004 xdot[k] u0[k] + 0.002 xdot[k] u1[k] + -0.015 xdot[k] ang[k]
    (xddot)[k+1] = 0.024 xdot[k] xddot[k]
    (yddot)[k+1] = 0.027 xdot[k] yddot[k] + 0.007 xdot[k] u0[k] + 0.003 xdot[k] ang[k]
        

        A11 = 0.703 + 0.009 * x_dot + 0.006 * yddot 
        A22 = 0.024 * x_dot
        A33 = 0.3 + 0.1 * x_dot
        #A33org = 0.027 * x_dot
        
        B11 =  0.004 * x_dot
        #B11 = 0.004 * x_dot

        #B12 = 0.004*x_dot
        B12 = 0.08*x_dot
        #B12org = 0.002 * x_dot

        B13 = -0.002 * x_dot
        #B13 = -0.015 * x_dot

        B31 = 0.2+ 0.002 * x_dot
#        B31 = 0.05 + 0.007 * x_dot
        #B31 = 0.007 * x_dot
        B33 =  0.026 * x_dot
        #B33org = 0.003 * x_dot
        '''

        '''
        (xdot)[k+1] = -0.011 1 + 0.999 xdot[k] + 0.094 u1[k] + -0.001 xddot[k]^2
        (xddot)[k+1] = 0.953 xddot[k] + 0.055 u1[k] + -0.002 xdot[k] u1[k] + -0.006 xddot[k]^2 + 0.001 yddot[k]^2
        (yddot)[k+1] = 0.981 yddot[k] + 0.002 xdot[k] u0[k] + 0.001 xdot[k] ang[k]

        A11 = -0.011/x_dot + 0.999 
        A12 = -0.001 * xddot
        A13 = 0
        A21 = 0
        A22 = 0.953 -0.006*xddot
        A23 = 0.001 * yddot
        A33 = 0.981 
        #A33org = 0.981 

        B12 = 0.094
        B22 = 0.055 - 0.002*x_dot
        B31 = 0.002*x_dot
        B33 = 0.001 * x_dot
        # Get the state space matrices for the control

        '''
        A11 = 0
        A12 = 0
        A13 = 0
        A21 = 0
        A22 = 0
        A23 = 0
        A31 = 0
        A32 = 0
        A33 = 0
        B11 = 0
        B12 = 0
        B13 = 0
        B21 = 0
        B22 = 0
        B23 = 0
        B31 = 0
        B32 = 0
        B33 = 0

        '''
        (xdot)[k+1] = -0.015 1 + 1.000 xdot[k] + 0.100 u1[k] + -0.001 xddot[k]^2
        (xddot)[k+1] = -0.001 xdot[k] + 0.951 xddot[k] + 0.070 u1[k] + -0.003 xdot[k] u1[k] + -0.005 u1[k]^2
        (yddot)[k+1] = 0.989 yddot[k] + 0.003 ang[k] + -0.001 xdot[k] yddot[k] + 0.002 xdot[k] u0[k] + 0.001 xdot[k] ang[k] + 0.002 yddot[k] ang[k]
        '''
        if self.version == 1:
         
            A11 = -0.015*1.001/(x_dot+0.001) + 1. 
            #A11 = -0.015*1.002/(x_dot+0.01) + 1. 
            #A11 = -0.015/(x_dot) + 1. 
            A12 = -0.001 * xddot
            A13 = 0
            A21 = -0.001
            A22 = 0.951
            A23 = 0
            A33 = self.constants['a33_ydd_0'] + self.constants['a33_ydd_1xv']*x_dot
            #A33 = 0.987 -0.001*x_dot
            #A33 = 0.989 -0.001*x_dot

            B12 = 0.1
            B22 = 0.070 -0.003*x_dot - 0.005*a
            B31 = self.constants['b31_u0_1xv'] *x_dot
            #B31 = 0.002*x_dot
            #B31org = 0.002*x_dot

            B33 = self.constants['b33_ang_0'] + self.constants['b33_ang_1xv'] *x_dot + self.constants['b33_ang_2ydd'] *yddot
            #B33 = 0.002 + 0.0005 *x_dot + 0.001*yddot
            #B33 = 0.0028 + 0.0008 *x_dot + 0.0018*yddot
            #B33 = 0.003 + 0.001 *x_dot + 0.002*yddot
            #B33org = 0.003 + 0.001 *x_dot + 0.002*yddot
 
        else: 
            ''' Model v5 - Sindy using mpc data 100x3 runs (Sep 15)
            
            (xdot)[k+1] = -0.007 1 + 0.999 xdot[k] + 0.002 xddot[k] + 0.092 u1[k] + -0.001 xddot[k]^2
	        (xddot)[k+1] = -0.002 xdot[k] + 0.890 xddot[k] + -0.003 yddot[k] + -0.005 u0[k] + 0.114 u1[k] + -0.003 xdot[k] u1[k] + -0.008 xddot[k]^2 + -0.002 xddot[k] yddot[k] + 0.007 	xddot[k] u0[k] + 0.001 yddot[k]^2 + 0.001 yddot[k] u0[k] + 0.002 u0[k]^2
	
        	(yddot)[k+1] = 0.943 yddot[k] + 0.204 u0[k] + 0.003 u1[k] + 0.079 ang[k] + -0.001 xdot[k] yddot[k] + -0.002 xdot[k] u0[k] + -0.002 xddot[k]^2 + -0.006 yddot[k]^2 + 0.010 yddot[k] u0[k] + 0.003 yddot[k] u1[k] + 0.017 u0[k] u1[k]
            '''    
            #print(f"State Space model 5 calc {self.constants['a33_ydd_0']}")
            A11 = -0.007*1.001/(x_dot+0.001)+0.999
            A12 = 0.002 - 0.001*xddot
            A13 = 0.

            A21 = -0.002
            A22 = 0.890 - 0.008*xddot -0.002*yddot
            A23 = -0.003 + 0.001*yddot

            A31 = 0
            A32 = -0.002*xddot
            
            A33 = self.constants['a33_ydd_0'] + self.constants['a33_ydd_1xv']*x_dot -self.constants['a33_ydd_2xydd']*yddot
            #A33org = 0.943 - 0.001*x_dot - 0.006*yddot

            B11 = 0
            B12 = self.constants['b12_u1_0'] 
            #B12org = 0.092
            B13 = 0

            B21 = -0.005 +0.007*xddot + 0.001*yddot + 0.002*delta
            B22 = 0.114 -0.003*x_dot
            B23 = 0.

            B31 = self.constants['b31_u0_0'] + self.constants['b31_u0_1xdv'] *xddot
            + self.constants['b31_u0_2xydd'] *yddot + self.constants['b31_u0_3xa']*a+self.constants['b31_u0_4xang']*ang
            #B31 = 0.104 -0.002*xddot + 0.01*yddot + 0.017*a #+ 0.02*ang
            #B31org = 0.204 -0.002*xddot + 0.01*yddot + 0.017*a
            
            B32 = 0.003 + 0.003*yddot
            
            B33 =  self.constants['b33_ang_0'] + self.constants['b33_ang_1xv']*x_dot
            #B33org = 0.079 
            #B33 = 0.75 - 0.001*x_dot



        A=np.array([[A11, A12, 0],[A21, A22, A23],[A31, A32,  A33]])
        B=np.array([[B11, B12, B13],[B21, B22, 0],[B31, B32, B33]]) 
        C=np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
        D=np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]]) 
        '''
        A=np.array([[A11, A12, 0, A14, 0, 0],[0, A22, 0, A24, 0, 0],[0, 0, 0, A34, 0, 0],\
        [0, A42, 0, A44, 0, 0],[A51, A52, 0, 0, 0, 0],[A61, A62, 0, 0, 0, 0]])
        B=np.array([[B11, B12],[B21, 0],[0, 0],[B41, 0],[0, 0],[0, 0]])
        '''
        # Discretise the system (forward Euler)
        Ad=A#np.identity(np.size(A,1))+Ts*A
        Bd=B#Ts*B
        Cd=C
        Dd=D

        return Ad, Bd, Cd, Dd

    def augmented_matrices(self, Ad, Bd, Cd, Dd):

        A_aug=np.concatenate((Ad,Bd),axis=1)
        temp1=np.zeros((np.size(Bd,1),np.size(Ad,1)))
        temp2=np.identity(np.size(Bd,1))
        temp=np.concatenate((temp1,temp2),axis=1)

        A_aug=np.concatenate((A_aug,temp),axis=0)
        B_aug=np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
        C_aug=np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)
        D_aug=Dd

        return A_aug, B_aug, C_aug, D_aug

    def mpc_simplification(self, Ad, Bd, Cd, Dd, hz, x_aug_t, du):
        '''This function creates the compact matrices for Model Predictive Control'''
        # db - double bar
        # dbt - double bar transpose
        # dc - double circumflex

        A_aug, B_aug, C_aug, D_aug=self.augmented_matrices(Ad, Bd, Cd, Dd)

        Q=self.constants['Q']
        S=self.constants['S']
        R=self.constants['R']
        Cf=self.constants['Cf']
        g=self.constants['g']
        m=self.constants['m']
        mju=self.constants['mju']
        lf=self.constants['lf']
        inputs=self.constants['inputs']

        ############################### Constraints #############################
        d_delta_max= 0.33# np.pi/300
        d_a_max=0.4#4#0.1
        d_delta_min=-0.33 #-np.pi/300
        d_a_min=-0.7#-7#0.1

        ub_global=np.zeros(inputs*hz)
        lb_global=np.zeros(inputs*hz)

        # Only works for 2 inputs
        for i in range(0,inputs*hz):
            if i%2==0:
                ub_global[i]=d_delta_max
                lb_global[i]=-d_delta_min
            else:
                ub_global[i]=d_a_max
                lb_global[i]=-d_a_min

        ub_global=ub_global[0:inputs*hz]
        lb_global=lb_global[0:inputs*hz]
        ublb_global=np.concatenate((ub_global,lb_global),axis=0)

        I_global=np.eye(inputs*hz)
        I_global_negative=-I_global
        I_mega_global=np.concatenate((I_global,I_global_negative),axis=0)

        y_asterisk_max_global=[]
        y_asterisk_min_global=[]
            
        #constraint on xdot, ydotdot
        C_asterisk=np.matrix('1 0 0 0 0 0;\
                        0 0 1 0 0 0;\
                        0 0 0 1 0 0;\
                        0 0 0 0 1 0'
                        )

        C_asterisk_global=np.zeros((np.size(C_asterisk,0)*hz,np.size(C_asterisk,1)*hz))

        #########################################################################

        CQC=np.matmul(np.transpose(C_aug),Q)
        CQC=np.matmul(CQC,C_aug)

        CSC=np.matmul(np.transpose(C_aug),S)
        CSC=np.matmul(CSC,C_aug)

        QC=np.matmul(Q,C_aug)
        SC=np.matmul(S,C_aug)

        Qdb=np.zeros((np.size(CQC,0)*hz,np.size(CQC,1)*hz))
        Tdb=np.zeros((np.size(QC,0)*hz,np.size(QC,1)*hz))
        Rdb=np.zeros((np.size(R,0)*hz,np.size(R,1)*hz))
        Cdb=np.zeros((np.size(B_aug,0)*hz,np.size(B_aug,1)*hz))
        Adc=np.zeros((np.size(A_aug,0)*hz,np.size(A_aug,1)))

        ######################### Advanced LPV ##################################
        A_product=A_aug
        states_predicted_aug=x_aug_t
        A_aug_collection=np.zeros((hz,np.size(A_aug,0),np.size(A_aug,1)))
        B_aug_collection=np.zeros((hz,np.size(B_aug,0),np.size(B_aug,1)))

        #########################################################################

        for i in range(0,hz):
            if i == hz-1:
                Qdb[np.size(CSC,0)*i:np.size(CSC,0)*i+CSC.shape[0],np.size(CSC,1)*i:np.size(CSC,1)*i+CSC.shape[1]]=CSC
                Tdb[np.size(SC,0)*i:np.size(SC,0)*i+SC.shape[0],np.size(SC,1)*i:np.size(SC,1)*i+SC.shape[1]]=SC
            else:
                Qdb[np.size(CQC,0)*i:np.size(CQC,0)*i+CQC.shape[0],np.size(CQC,1)*i:np.size(CQC,1)*i+CQC.shape[1]]=CQC
                Tdb[np.size(QC,0)*i:np.size(QC,0)*i+QC.shape[0],np.size(QC,1)*i:np.size(QC,1)*i+QC.shape[1]]=QC

            Rdb[np.size(R,0)*i:np.size(R,0)*i+R.shape[0],np.size(R,1)*i:np.size(R,1)*i+R.shape[1]]=R

            ########################### Advanced LPV ############################
            Adc[np.size(A_aug,0)*i:np.size(A_aug,0)*i+A_aug.shape[0],0:0+A_aug.shape[1]]=A_product
            A_aug_collection[i][:][:]=A_aug
            B_aug_collection[i][:][:]=B_aug
            #####################################################################

            ######################## Constraints ################################
            x_dot_max=50
            delta_max=2
            a_max =5 
            a_min = -8
            
            x_dot_min=0.
            delta_min=-2. #-np.pi/6
            a_min=-4+(states_predicted_aug[2][0]+mju*m*g)/m
            a_max=1+(states_predicted_aug[2][0]+mju*m*g)/m

            yddot_max = 4.0
            yddot_min = -4.0

            
            y_asterisk_max=np.array([x_dot_max,yddot_max, delta_max,a_max])
            y_asterisk_min=np.array([x_dot_min,yddot_min, delta_min, a_min])

            y_asterisk_max_global=np.concatenate((y_asterisk_max_global,y_asterisk_max),axis=0)
            y_asterisk_min_global=np.concatenate((y_asterisk_min_global,y_asterisk_min),axis=0)
            C_asterisk_global[np.size(C_asterisk,0)*i:np.size(C_asterisk,0)*i+C_asterisk.shape[0],np.size(C_asterisk,1)*i:np.size(C_asterisk,1)*i+C_asterisk.shape[1]]=C_asterisk


            #####################################################################

            ######################### Advanced LPV ##############################
            if i<hz-1:
                du1=du[inputs*(i+1)][0]
                du2=du[inputs*(i+1)+1][0]
                du3=du[inputs*(i+1)+2][0]
                states_predicted_aug=np.matmul(A_aug,states_predicted_aug)+np.matmul(B_aug,np.transpose([[du1,du2, du3]]))
                states_predicted = states_predicted_aug.T[0]
                delta_predicted=states_predicted_aug[3][0]
                a_predicted=states_predicted_aug[4][0]
                Ad, Bd, Cd, Dd=self.state_space(states_predicted,delta_predicted,a_predicted, du3 )
                A_aug, B_aug, C_aug, D_aug=self.augmented_matrices(Ad, Bd, Cd, Dd)
                A_product=np.matmul(A_aug,A_product)

        for i in range(0,hz):
            for j in range(0,hz):
                if j<=i:
                    AB_product=np.eye(np.shape(A_aug)[0])
                    for ii in range(i,j-1,-1):
                        if ii>j:
                            AB_product=np.matmul(AB_product,A_aug_collection[ii][:][:])
                        else:
                            AB_product=np.matmul(AB_product,B_aug_collection[ii][:][:])
                    Cdb[np.size(B_aug,0)*i:np.size(B_aug,0)*i+B_aug.shape[0],np.size(B_aug,1)*j:np.size(B_aug,1)*j+B_aug.shape[1]]=AB_product

        #########################################################################

        ####################### Constraints #####################################

        Cdb_constraints=np.matmul(C_asterisk_global,Cdb)
        Cdb_constraints_negative=-Cdb_constraints
        Cdb_constraints_global=np.concatenate((Cdb_constraints,Cdb_constraints_negative),axis=0)

        Adc_constraints=np.matmul(C_asterisk_global,Adc)
        Adc_constraints_x0=np.transpose(np.matmul(Adc_constraints,x_aug_t))[0]
        y_max_Adc_difference=y_asterisk_max_global-Adc_constraints_x0
        y_min_Adc_difference=-y_asterisk_min_global+Adc_constraints_x0
        y_Adc_difference_global=np.concatenate((y_max_Adc_difference,y_min_Adc_difference),axis=0)

        G=np.concatenate((I_mega_global,Cdb_constraints_global),axis=0)
        ht=np.concatenate((ublb_global,y_Adc_difference_global),axis=0)

        #######################################################################

        Hdb=np.matmul(np.transpose(Cdb),Qdb)
        Hdb=np.matmul(Hdb,Cdb)+Rdb

        temp=np.matmul(np.transpose(Adc),Qdb)
        temp=np.matmul(temp,Cdb)

        temp2=np.matmul(-Tdb,Cdb)
        Fdbt=np.concatenate((temp,temp2),axis=0)

        return Hdb,Fdbt,Cdb,Adc,G,ht

    def open_loop_new_states(self,states,delta,a,ang):
        '''This function computes the new state vector for one sample time later'''

        # Get the necessary constants
        g=self.constants['g']
        m=self.constants['m']
        Iz=self.constants['Iz']
        Cf=self.constants['Cf']
        Cr=self.constants['Cr']
        lf=self.constants['lf']
        lr=self.constants['lr']
        Ts=self.constants['Ts']
        mju=self.constants['mju']

        current_states=states
        new_states=current_states

        xdot=current_states[0]
        xddot=current_states[1]
        yddot=current_states[2]


        xdot_kp1 = 0.703* xdot + 0.009* xdot**2 + 0.006 *xdot*yddot #
        #+ 0.004 *xdot[k] *u0[k] + 0.002 xdot[k] u1[k] + -0.015 xdot[k] ang[k]
        xddot_kp1 = 0.024 *xdot* xddot
        yddot_kp1 = 0.027 *xdot *yddot #+ 0.007 *xdot[k] u0[k] + 0.003 xdot[k] ang[k]

        # Take the last states
        new_states[0]=xdot_kp1# x_dot
        new_states[1]=xddot_kp1#y_dot
        new_states[2]=yddot_kp1#psi

        return new_states#,x_dot_dot,y_dot_dot,psi_dot_dot

    def closed_loop_new_states(self,states,u0,u1,ang):
        '''This function computes the new state vector for one sample time later'''

        # Get the necessary constants
        g=self.constants['g']
        m=self.constants['m']
        Iz=self.constants['Iz']
        Cf=self.constants['Cf']
        Cr=self.constants['Cr']
        lf=self.constants['lf']
        lr=self.constants['lr']
        Ts=self.constants['Ts']
        mju=self.constants['mju']

        current_states=states
        new_states=current_states

        xdot=current_states[0]
        xddot=current_states[1]
        yddot=current_states[2]


        A,B,C,D = self.state_space(states,u0,u1,ang)
        new_states = np.matmul(A, states) + np.matmul(B, np.array([u0,u1,ang]))
        print(f"Mat Mul: A@ {states} +B: {new_states}")
        '''
        xdot_kp1 = 0.703* xdot + 0.009* xdot**2 + 0.006 *xdot*yddot+ 0.004 *xdot *u0 + 0.002 *xdot *u1 + -0.015 *xdot* ang
        xddot_kp1 = 0.024 *xdot* xddot
        yddot_kp1 = 0.027 *xdot *yddot + 0.007 *xdot* u0 + 0.003 *xdot* ang

        # Take the last states
        new_states[0]=xdot_kp1# x_dot
        new_states[1]=xddot_kp1#y_dot
        new_states[2]=yddot_kp1#psi
        print(f"cloed loop mul: A@ states +B: {new_states}")
        '''
        return new_states#,x_dot_dot,y_dot_dot,psi_dot_dot

