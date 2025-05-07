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
            sys.stdout.flush()

        def Initialize(self):
            self.start_time = time.time()
            super().Initialize()
            sys.stdout.flush()
            self.main_model_part = self.model.GetModelPart("Structure")
            self.modes = np.load("rom_data/RightBasisMatrix.npy").T
            self.step = 0

        def InitializeSolutionStep(self):
            super().InitializeSolutionStep()
            self.node_index = 0
            for node in self.main_model_part.Nodes:
                node.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X, self.modes[self.step,self.node_index])
                node.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y, self.modes[self.step,self.node_index+1])
                node.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Z, self.modes[self.step,self.node_index+2])
                node.Fix(KratosMultiphysics.DISPLACEMENT_X)
                node.Fix(KratosMultiphysics.DISPLACEMENT_Y)
                node.Fix(KratosMultiphysics.DISPLACEMENT_Z)
                self.node_index += 3
            self.step += 1
        

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()
        
        def Finalize(self):
            super().Finalize()

    return AnalysisStageWithFlush(global_model, parameters)

if __name__ == "__main__":

    with open("ProjectParameters_modes.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    global_model = KratosMultiphysics.Model()
    simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters)
    simulation.Run()
