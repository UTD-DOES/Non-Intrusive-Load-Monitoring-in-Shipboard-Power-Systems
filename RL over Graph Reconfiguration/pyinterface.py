"""
Example script to test the running of simulink ship model using python
"""
import matlab.engine
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator

import numpy as np
import gym
from gym import spaces


eng = matlab.engine.start_matlab() # creating an instance of the engine
print("Matlab connection established")


# Simulink project load
proj_path=r"C:\Users\UTD\Box\New folder (sxs190214@utdallas.edu)\ONR\reconfiguration\Roshni Environment\Two_Zone_MVDC.prj"
eng.eval("proj = simulinkproject('{}')".format(proj_path),nargout=0) # Directly input simulink model name into script


SimModel= "Two_Zone_MVDC_PSolver_R2020b" #name of the simulink model

# MatModel= "Two_Zone_MVDC_Ideal"      #matlab file which executes the simulink model
# Execute the simulink model using the .m file
# eng.Two_Zone_MVDC_Ideal(nargout=0) # to open the simulink model using matlab script(too time consuming)


# Directly execute the simulink model using script
eng.eval("model = '{}'".format(SimModel),nargout=0) # Directly input simulink model name into script
eng.eval("load_system(model)",nargout=0)  #Loading the simulink model
# eng.get_param('Two_Zone_MVDC_PSolver/PGM2/Solver Configuration','DialogParameters') # to get a list of all parameters
print("\n Solver Choice initial:")
print(eng.get_param('{}/Subsystem4/Solver Configuration'.format(SimModel),'LocalSolverChoice'))
# print("\n Changing solver type......")
# eng.set_param('{}/Subsystem4/Solver Configuration'.format(SimModel),'LocalSolverChoice','NE_BACKWARD_EULER_ADVANCER',nargout=0)
# print("\n New solver choice is:")
# print(eng.get_param('{}/Subsystem4/Solver Configuration'.format(SimModel),'LocalSolverChoice'))
eng.set_param(SimModel,'StopTime','0.1',nargout=0)
eng.set_param(SimModel,'SimulationCommand','Start',nargout=0)
# eng.eval("simOut=sim('{}')".format(SimModel),nargout=0)
while eng.get_param(SimModel, 'SimulationStatus') == 'running':
      eng.set_param(SimModel, 'SimulationCommand','continue',nargout=0)
      
# Read the control variables
#---- Use this to extract base/original configuration
model_vars=eng.eval("who") # list of all variables in the current workspace
ModelVars_dict = {}
for v in model_vars:
    ModelVars_dict [v] = eng.workspace[v]

# Read the state variables (get_state function in RL)
# out_signal_components= np.array(eng.eval("get(logsout)")) # an array of signal measurements
out_components = np.array(eng.eval("get(logsout)"))
for indx in range(len(out_components)):
    comp_Name = eng.eval("logsout{" + str(indx+1) + "}.Name")
    if comp_Name == 'GEN1':
        Gen1_t = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.P.Time"))
        Gen1_P = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.P.Data"))       
    
    if comp_Name == 'GEN2':
        Gen2_t = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.P.Time"))
        Gen2_P = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.P.Data"))       

    if comp_Name == 'VDC1':
        VDC1_t = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.Time"))
        VDC1= np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.Data"))            

    if comp_Name == 'VDC2':
        VDC2_t = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.Time"))
        VDC2= np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.Data"))    
   
Gen1_reward = np.sum(np.abs(1 - Gen1_P))
Gen2_reward = np.sum(np.abs(1 - Gen2_P))
reward = Gen1_reward + Gen2_reward
print(reward)



# eng.evalc("GABDA=0;")

# eng.set_param(SimModel,'StopTime','1',nargout=0)
# eng.set_param(SimModel,'SimulationCommand','Start',nargout=0)
# # eng.eval("simOut=sim('{}')".format(SimModel),nargout=0)
# while eng.get_param(SimModel, 'SimulationStatus') == 'running':
#       eng.set_param(SimModel, 'SimulationCommand','continue',nargout=0)
      

# for indx in range(len(out_components)):
#     comp_Name = eng.eval("logsout{" + str(indx+1) + "}.Name")
#     if comp_Name == 'GEN1':
#         Gen1_t = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.P.Time"))
#         Gen1_P = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.P.Data"))       
    
#     if comp_Name == 'GEN2':
#         Gen2_t = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.P.Time"))
#         Gen2_P = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.P.Data"))       

#     if comp_Name == 'VDC1':
#         VDC1_t = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.Time"))
#         VDC1= np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.Data"))            

#     if comp_Name == 'VDC2':
#         VDC2_t = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.Time"))
#         VDC2= np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.Data"))    
        
# Gen1_reward = np.sum(np.abs(1 - Gen1_P))
# Gen2_reward = np.sum(np.abs(1 - Gen2_P))
# reward = Gen1_reward + Gen2_reward
# print(reward)        
        
        
        
        
        
# eng.evalc("GABDA=0;")
# eng.evalc("GBBDB=0;")

# eng.set_param(SimModel,'StopTime','1',nargout=0)
# eng.set_param(SimModel,'SimulationCommand','Start',nargout=0)
# # eng.eval("simOut=sim('{}')".format(SimModel),nargout=0)
# while eng.get_param(SimModel, 'SimulationStatus') == 'running':
#       eng.set_param(SimModel, 'SimulationCommand','continue',nargout=0)
      

# for indx in range(len(out_components)):
#     comp_Name = eng.eval("logsout{" + str(indx+1) + "}.Name")
#     if comp_Name == 'GEN1':
#         Gen1_t = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.P.Time"))
#         Gen1_P = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.P.Data"))       
    
#     if comp_Name == 'GEN2':
#         Gen2_t = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.P.Time"))
#         Gen2_P = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.P.Data"))       

#     if comp_Name == 'VDC1':
#         VDC1_t = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.Time"))
#         VDC1= np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.Data"))            

#     if comp_Name == 'VDC2':
#         VDC2_t = np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.Time"))
#         VDC2= np.array(eng.eval("logsout{" + str(indx+1) + "}.Values.Data"))    
        
# Gen1_reward = np.sum(np.abs(1 - Gen1_P))
# Gen2_reward = np.sum(np.abs(1 - Gen2_P))
# reward = Gen1_reward + Gen2_reward
# print(reward)





#####################################################################################



class ShipEnvironment(gym.Env):
    
    def __init__(self, proj_path, SimModel):
        
        print("Initializing Ship env")
        super(ShipEnvironment, self).__init__()
        self.proj_path = proj_path
        self.SimModel = SimModel
        self.eng = matlab.engine.start_matlab()
        self.action_space = spaces.Discrete(256)  # Example: 0: Do nothing, 1: Increase speed, 2: Decrease speed
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)  # Example: [Gen1_power, Gen2_power, VDC1_voltage, VDC2_voltage]
        print("Env Initialized")
        
        
        
        
    def reset(self):
        self.eng.eval("proj = simulinkproject('{}')".format(self.proj_path), nargout=0)
        self.eng.eval("model = '{}'".format(self.SimModel), nargout=0)
        self.eng.eval("load_system(model)", nargout=0)
        self.eng.set_param(self.SimModel, 'StopTime', '0.1', nargout=0)
        self.eng.set_param(self.SimModel, 'SimulationCommand', 'Start', nargout=0)
        while self.eng.get_param(self.SimModel, 'SimulationStatus') == 'running':
            self.eng.set_param(self.SimModel, 'SimulationCommand', 'continue', nargout=0)
        self.state = self._get_state()
        return self.state

    def step(self, action):
        action_str = format(action, '08b')
        self.eng.evalc(f"GBBDB={action_str[-1]};")
        self.eng.evalc(f"SDDC={action_str[-1]};")

        self.eng.evalc(f"GBBDA={action_str[-2]};")
        self.eng.evalc(f"SCDC={action_str[-2]};")

        self.eng.evalc(f"GABDA={action_str[-3]};")
        self.eng.evalc(f"SADC={action_str[-3]};")
        
        self.eng.evalc(f"GABDB={action_str[-4]};")
        self.eng.evalc(f"SBDC={action_str[-4]};")
        
        self.eng.evalc(f"SADA={action_str[-5]};")
        self.eng.evalc(f"SBDA={action_str[-5]};")
        
        self.eng.evalc(f"SCDB={action_str[-6]};")
        self.eng.evalc(f"SDDB={action_str[-6]};")
        
        self.eng.evalc(f"SADB={action_str[-7]};")
        self.eng.evalc(f"SCDA={action_str[-7]};")
        
        self.eng.evalc(f"SBDB={action_str[-8]};")
        self.eng.evalc(f"SDDA={action_str[-8]};")


        # if action == 1:
        #     # Increase speed
        #     self.eng.evalc("GABDA=0;")
        #     self.eng.evalc("GBBDB=1;")
        # elif action == 2:
        #     # Decrease speed
        #     self.eng.evalc("GABDA=0;")
        #     self.eng.evalc("GBBDB=0;")
        # elif action == 0:
        #     # Do nothing
        #     self.eng.evalc("GABDA=1;")
        #     self.eng.evalc("GBBDB=1;")
    
        # Start simulation
        self.eng.set_param(self.SimModel, 'StopTime', '0.1', nargout=0)
        self.eng.set_param(self.SimModel, 'SimulationCommand', 'Start', nargout=0)
        
        # Run simulation step by step
        #self.eng.set_param(self.SimModel, 'SimulationCommand', 'step', nargout=0)
    
        # Wait until the simulation finishes
        while self.eng.get_param(self.SimModel, 'SimulationStatus') == 'running':
            self.eng.set_param(self.SimModel, 'SimulationCommand', 'continue', nargout=0)
    
        # Update the state with new values
        self.state = self._get_state()

        # Calculate reward
        Gen1_reward = np.abs(self.state[0,-1,0])
        Gen2_reward = np.abs(self.state[1,-1,0])
        V1_reward = 12000 - np.abs(self.state[2,-1,0])
        V2_reward = 12000 - np.abs(self.state[3,-1,0])
        PL1_reward = 1440000 - np.abs(self.state[4,-1,0])
        PL2_reward = 1440000 - np.abs(self.state[5,-1,0])
        #reward = Gen1_reward + Gen2_reward
        reward = PL1_reward + PL2_reward
        done = self._is_done()
    
   
    
        return self.state, -reward, done, {}



    def _get_state(self):
        # Fetch relevant data from simulation
        out_components = np.array(self.eng.eval("get(logsout)"))
        for indx in range(len(out_components)):
            comp_Name = self.eng.eval("logsout{" + str(indx+1) + "}.Name")
            if comp_Name == 'GEN1':
                Gen1_t = np.array(self.eng.eval("logsout{" + str(indx+1) + "}.Values.P.Time"))
                Gen1_P = np.array(self.eng.eval("logsout{" + str(indx+1) + "}.Values.P.Data"))       
            
            if comp_Name == 'GEN2':
                Gen2_t = np.array(self.eng.eval("logsout{" + str(indx+1) + "}.Values.P.Time"))
                Gen2_P = np.array(self.eng.eval("logsout{" + str(indx+1) + "}.Values.P.Data"))       

            if comp_Name == 'VDC1':
                VDC1_t = np.array(self.eng.eval("logsout{" + str(indx+1) + "}.Values.Time"))
                VDC1= np.array(self.eng.eval("logsout{" + str(indx+1) + "}.Values.Data"))            

            if comp_Name == 'VDC2':
                VDC2_t = np.array(self.eng.eval("logsout{" + str(indx+1) + "}.Values.Time"))
                VDC2= np.array(self.eng.eval("logsout{" + str(indx+1) + "}.Values.Data"))  
                
            if comp_Name == 'P_Load1':
                PL1_t = np.array(self.eng.eval("logsout{" + str(indx+1) + "}.Values.Time"))
                PL1= np.array(self.eng.eval("logsout{" + str(indx+1) + "}.Values.Data"))  
                
            if comp_Name == 'P_Load2':
                PL2_t = np.array(self.eng.eval("logsout{" + str(indx+1) + "}.Values.Time"))
                PL2= np.array(self.eng.eval("logsout{" + str(indx+1) + "}.Values.Data"))  
                
        return np.array([Gen1_P, Gen2_P, VDC1, VDC2, PL1, PL2])

    def _calculate_reward(self):
        Gen1_reward = np.sum(np.abs(1 - self.state[0]))
        Gen2_reward = np.sum(np.abs(1 - self.state[1]))
       
        reward = Gen1_reward + Gen2_reward
        return -reward  # Negative reward as we want to minimize the absolute difference

    def _is_done(self):
        # Example termination condition
        if np.abs(self.state[4,-1,0]) < 1000000 or np.abs(self.state[5,-1,0]) < 1000000:
            return True  

    def close(self):
        self.eng.quit()



# Example usage

# env = ShipEnvironment(proj_path, SimModel)
# obs = env.reset()



# for _ in range(300):
#      action = env.action_space.sample()  # Sample random action
#      obs, reward, done, _ = env.step(action)
#      print(f"reward : {reward}, action: {action}, done: {done}")

#      if done:
#          print("Done!!!")
#          break



#env.close()

