import sys
import time
import importlib
import numpy as np
import time

import KratosMultiphysics
from KratosMultiphysics.assign_scalar_variable_to_conditions_process import AssignScalarVariableToConditionsProcess

def CreateAnalysisStageWithFlushInstance(cls, global_model, parameters):
    class AnalysisStageWithFlush(cls):

        def __init__(self, model,project_parameters, flush_frequency=10.0):
            super().__init__(model,project_parameters)
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            sys.stdout.flush()

        def Initialize(self):
            self.start_time = time.time()
            super().Initialize()
            sys.stdout.flush()
            self.main_model_part = self.model.GetModelPart("Structure")
            self.snapshots_matrix = []

        def InitializeSolutionStep(self):
            super().InitializeSolutionStep()
        
        def ApplyBoundaryConditions(self):
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
            t = self.time
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

            AssignScalarVariableToConditionsProcess(self.model, pressure_settings).ExecuteInitializeSolutionStep()

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()
            snapshot = []
            for node in self.main_model_part.Nodes:
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y))
                snapshot.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Z))
            self.snapshots_matrix.append(snapshot)

            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now
        
        def Finalize(self):
            self.end_time = time.time()
            print(f"Elapsed time: {self.end_time - self.start_time:.2f} s")
            super().Finalize()
            np.save("FOM.npy",self.snapshots_matrix)

    return AnalysisStageWithFlush(global_model, parameters)

if __name__ == "__main__":

    with open("ProjectParameters_modified.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    global_model = KratosMultiphysics.Model()
    simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters)
    simulation.Run()
