:Begin:
:Function:       Get42
:Pattern:        Get42[]
:Arguments:      {  }
:ArgumentTypes:  {  }
:ReturnType:     Integer
:End:

:Begin:
:Function:       RunTestsM
:Pattern:        RunTestsM[]
:Arguments:      {  }
:ArgumentTypes:  {  }
:ReturnType:     Integer
:End:

:Evaluate: $oldContextPath = $ContextPath; $ContextPath = {"System`", "Global`"}; (* these are the only dependencies *)
:Evaluate: Begin@"InfiniTAM2`Private`" (* create everythin in InfiniTAM2`Private`* *)
:Evaluate: UnprotectClearAll@"InfiniTAM2`Private`*" (* create everythin in InfiniTAM`Private`* *)


:Evaluate: Protect@"InfiniTAM2`Private`*"
:Evaluate: End[] 
:Evaluate: $ContextPath = $oldContextPath
