import KratosMultiphysics

import KratosMultiphysics.RomApplication.rom_testing_utilities as rom_testing_utilities
from KratosMultiphysics.assign_scalar_variable_to_conditions_process import AssignScalarVariableToConditionsProcess
import numpy as np
import time
"""
For user-scripting it is intended that a new class is derived
from StructuralMechanicsAnalysis to do modifications
"""

if __name__ == "__main__":

    with open("ProjectParameters_modified_rom.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    model = KratosMultiphysics.Model()
    dummy = rom_testing_utilities.SetUpSimulationInstance(model,parameters)
    
    class DummyAnalysis(type(dummy)):

        def Initialize(cls):
            cls.start_time = time.time()
            super().Initialize()
            cls.main_model_part = cls.model.GetModelPart("Structure")
            cls.snapshots_matrix = []
            cls.tip_node_magnitude_vector = []

        def ApplyBoundaryConditions(cls):
            super().ApplyBoundaryConditions()

            # --- Training pulse schedule ---
            pulse_schedule = [
                (0.0, 5e2),
                (2.0, 3.5e2),
                (4.0, 6.5e2),
                (6.0, 2.5e2)
            ]
            pulse_duration = 2.0
            L = 43.2  # blade length in meters
            x = L     # evaluating at tip (for global scalar value, x/L = 1)

            # --- Compute current pressure based on time ---
            t = cls.time
            pressure = 0.0
            for t_start, amplitude in pulse_schedule:
                t_end = t_start + pulse_duration
                if t_start <= t < t_end:
                    t_local = t - t_start
                    pressure = amplitude * (x / L)**2 * np.sin(np.pi * t_local / pulse_duration)**2
                    break

            # --- Define and assign scalar pressure value to surface ---
            pressure_settings = KratosMultiphysics.Parameters("""
            {
                "model_part_name" : "Structure.SurfacePressure3D_Pressure",
                "variable_name"   : "POSITIVE_FACE_PRESSURE",
                "interval"        : [0.0, "End"]
            }
            """)
            pressure_settings.AddEmptyValue("value").SetDouble(pressure)

            AssignScalarVariableToConditionsProcess(cls.model, pressure_settings).ExecuteInitializeSolutionStep()

        def FinalizeSolutionStep(cls):
            super().FinalizeSolutionStep()
            snapshot = []
            for node in cls.main_model_part.Nodes:
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Z))
            cls.snapshots_matrix.append(snapshot)
            tip_node = cls.main_model_part.GetNode(1)
            tip_node_x = tip_node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X)
            tip_node_y = tip_node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y)
            tip_node_z = tip_node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Z)
            disp_vector = np.array([tip_node_x, tip_node_y, tip_node_z])
            cls.tip_node_magnitude_vector.append(np.linalg.norm(disp_vector))
        
        def Finalize(cls):
            cls.end_time = time.time()
            print(f"Elapsed time: {cls.end_time - cls.start_time:.2f} s")
            super().Finalize()
            np.save("ROM_test.npy",cls.snapshots_matrix)
            np.save("tip_node_magnitude_vector_rom.npy",cls.tip_node_magnitude_vector)
        
    simulation = DummyAnalysis(model, parameters)

    # Run test case
    simulation.Run()

    fom = np.load("FOM_test.npy")
    rom = np.load("ROM_test.npy")

    error = 100*np.linalg.norm(rom-fom)/np.linalg.norm(fom)
    print(f"The error was of: {error:.2f}%")
