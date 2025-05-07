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

        def ApplyBoundaryConditions(cls):
            super().ApplyBoundaryConditions()

            # --- Training pulse schedule ---
            pulse_schedule = [
                (0.0, 5e2),
                (2.0, 4e2),
                (4.0, 6e2),
                (6.0, 3e2)
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
        
        def Finalize(cls):
            cls.end_time = time.time()
            print(f"Elapsed time: {cls.end_time - cls.start_time:.2f} s")
            super().Finalize()
            np.save("ROM.npy",cls.snapshots_matrix)
        
    simulation = DummyAnalysis(model, parameters)

    # Run test case
    simulation.Run()

    fom = np.load("FOM.npy")
    rom = np.load("ROM.npy")

    error = 100*np.linalg.norm(rom-fom)/np.linalg.norm(fom)
    print(f"The error was of: {error:.2f}%")