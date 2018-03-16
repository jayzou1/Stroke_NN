import numpy as np
import pandas as pd 
import os

def parse_data(chunk_size = 500, num_sensors = 5, num_acc = 9):
    path = './Data/'
    new_path = './processed_feature_label/'
    p_id = {}
    p_sides ={
    '1' : 'right',
    '2' : 'right',
    '3' : 'left',
    '4' : 'right',
    '5' : 'left',
    '6' : 'right',
    '7' : 'left',
    '8' : 'left',
    '9' : 'right',
    '10' : 'right',
    '11' : 'left',
    '12' : 'left',
    '14' : 'right',
    '15' : 'right',
    '16' : 'right',
    '17' : 'left',
    '18' : 'left',
    '19' : 'right',
    '20' : 'left'
    }
    for patients in next(os.walk(path))[1]:
        p_id[patients] = (path + patients + '/')
    
    movements = []
    movement_labels = []
    dfs = {}
    for id, filepath in p_id.items():
        f = pd.read_csv(filepath + "AdjustedJointAngles/adjusted_joint_angles.csv")
        l = pd.read_csv(filepath + "AdjustedLabel/label.csv")
        a = pd.read_csv(filepath + "PosVelAcc/pos_vel_acc.csv")
        
        time = list(zip(l["reaching_begin"].values, l["reaching_end"].values, l["retracting_begin"], l["retracting_end"]))
        labels = list(zip(l["FAS_reaching"], l["coordination_reaching"], l["e-coaching_reaching"], l["FAS_retracting"], l["coordination_retracting"], l["e-coaching_retracting"]))
        
        if p_sides[id] == "right":
            f = f[f.columns[::-1]]
        
        f.columns = ["affected_elbow",
                     "affected_shoulder",
                     "trunk",
                     "unaffected_shoulder",
                     "unaffected_elbow"]
        a.columns = ["position x",
                     "position y",
                     "position z",
                     "velocity x",
                     "velocity y",
                     "velocity z",
                     "acceleration x",
                     "acceleration y",
                     "acceleration z"]

        f = f.iloc[:, :num_sensors]
        a = a.iloc[:, len(a.columns) - num_acc:]

        for movement, label in zip(time, labels):
            start_begin, reach_end, retract_begin, retract_end = movement;
            label_list = np.asarray(label)
            slice = f[start_begin:retract_end+1]
            aslice = a[start_begin:retract_end+1]
            
            if len(slice) == 0 or len(aslice) == 0:
                continue
            smaller_slice = np.minimum(len(slice), len(aslice))
            if smaller_slice < chunk_size:
                chunk_miss = np.zeros((chunk_size - smaller_slice, num_sensors + len(aslice.columns)))
                movement = np.concatenate((slice[:smaller_slice].values, aslice[:smaller_slice].values), axis = 1)
                movement = np.vstack((movement, chunk_miss))
            else:
                movement = np.concatenate((slice[:chunk_size].values, aslice[:chunk_size].values), axis = 1)

            movements.append(movement)
            movement_labels.append(label_list)

    np.savez(new_path + str(num_sensors) + "_" + str(num_acc) + "_" + "data", features = movements, labels = movement_labels)

if __name__ == "__main__":
   parse_data()
