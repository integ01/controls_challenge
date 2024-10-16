
import numpy as np
import pandas as pd

dt=0.1

def calc_long_u_accel(df):
    area = 0.74 #m**2
    drag_coef =  0.35
    air_density = 1.225 #kg/m**3
    vel_square = df['v_ego']**2 #m/sec
    car_weight = 1400 #kg
    car_mu=0.02 # friction coefficient
    a_drag = 0.5*air_density*drag_coef*area*vel_square.values/car_weight
    a_fric = car_mu * 9.81 
    #print( a_drag)
    #print(a_fric)
    #print(type(a_drag))
    long_accel = a_drag + df['a_ego'].values + a_fric# - df['actual_lataccel']

    return long_accel

turnRadiusByAngle = [ (1,	154.69),
                    (2,	77.317),
                    (3,	51.52),
                    (4,	38.61),
                    (5,	30.86),
                    (6,	25.69),
                    (7,	21.99),
                    (8,	19.21),
                    (9,	17.05),
                    (10,15.31)]

invRadToAngel_intercept=-0.00018521697668676118
invRadToAngle_slope=0.00653436

def calc_ydot_integrator(df):
    y_dot_i = np.zeros(len(df))
    acclat =  df['actual_lataccel'].values
    y_dot_i[1:]= np.cumsum(acclat[:-1])*dt
    return y_dot_i

def calc_ratio_yddot_xddot(df):
    phi_dot_i = np.zeros(len(df))
    acclat =  df['actual_lataccel'].values
    acclong = df['a_ego'].values 
    phi_dotdot = acclat/(acclong+0.00001)
#    phi_dot_i[1:]= np.cumsum(phi_dotdot[:-1])*dt
    return phi_dot_i

def calc_phi_integrator(df):
    phi_dot_i = np.zeros(len(df))
    acclat =  df['actual_lataccel'].values
    acclong = df['a_ego'].values 
    phi_dotdot = (acclat**2 + acclong**2)**0.5
    phi_dot_i[1:]= np.cumsum(phi_dotdot[:-1])*dt
    return phi_dot_i
'''
## Data fields -     roll_lataccel      v_ego     a_ego  target_lataccel  steer_command  actual_lataccel
'''
def get_sim_run_data2( data_paths: list[str], st_idx=100, seq_len=100,dt=0.1, rnd=False) -> pd.DataFrame:
    X_train = []
    X_control = []
    st_idx_ = st_idx
    feature_names=[ "xdot", "xddot", "yddot", "u0", "u1","ang"]
    for i, data_path in enumerate(data_paths):
        if rnd:
            st_idx = np.random.randint(st_idx_,st_idx_+100)
        df = pd.read_csv(data_path)
        gas_brake_estim = calc_long_u_accel(df)
        X_train_r = np.stack((
            df['v_ego'].values[st_idx:st_idx+seq_len],
            df['a_ego'].values[st_idx:st_idx+seq_len], 
            df['actual_lataccel'].values[st_idx:st_idx+seq_len]),axis=-1)
        X_control_r  = np.stack((
            df['steer_command'][st_idx:st_idx+seq_len].values,
            gas_brake_estim[st_idx:st_idx+seq_len],
            #V2 Add target
            #df['target_lataccel'].values[st_idx:st_idx+seq_len]- df['steer_command'][st_idx:st_idx+seq_len].values,
            df['roll_lataccel'].values[st_idx:st_idx+seq_len]
            ),axis=-1)
        X_train.append(X_train_r)
        X_control.append(X_control_r)
        #if i == 0:
        #    X_train = X_train_r
        #    X_control = X_control_r
        #else:
        #    X_train = np.vstack([X_train,X_train_r])
        #    X_control = np.vstack([X_control,X_control_r])
    return X_train,X_control,feature_names




def get_sim_run_data( data_paths: list[str], st_idx=100, seq_len=100,dt=0.1) -> pd.DataFrame:
    X_train = []
    X_control = []
    feature_names=[ "xdot", "xddot", "ydot", "yddot","u0", "u1","ang"]
    for i, data_path in enumerate(data_paths):
        df = pd.read_csv(data_path)
        gas_brake_estim = calc_long_u_accel(df)
        ydot_i = calc_ratio_yddot_xddot(df)
        X_train_r = np.stack((
            df['v_ego'].values[st_idx:st_idx+seq_len],
            df['a_ego'].values[st_idx:st_idx+seq_len], 
            ydot_i[st_idx:st_idx+seq_len],
            df['actual_lataccel'].values[st_idx:st_idx+seq_len]),axis=-1)
        X_control_r  = np.stack((df['steer_command'][st_idx:st_idx+seq_len].values,
            gas_brake_estim[st_idx:st_idx+seq_len],
            df['roll_lataccel'].values[st_idx:st_idx+seq_len]
            ),axis=-1)
        X_train.append(X_train_r)
        X_control.append(X_control_r)
        #if i == 0:
        #    X_train = X_train_r
        #    X_control = X_control_r
        #else:
        #    X_train = np.vstack([X_train,X_train_r])
        #    X_control = np.vstack([X_control,X_control_r])
    return X_train,X_control,feature_names

