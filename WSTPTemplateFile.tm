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



:Begin:
:Function:       createScene
:Pattern:        createScene[voxelSize_Real]
:Arguments:      { voxelSize }
:ArgumentTypes:  { Real }
:ReturnType:     Integer
:End:


:Begin:
:Function:       getSceneVoxelSize
:Pattern:        getSceneVoxelSize[id_Integer?NonNegative]
:Arguments:      { id }
:ArgumentTypes:  { Integer }
:ReturnType:     Real
:End:


:Begin:
:Function:       sceneExistsQ
:Pattern:        sceneExistsQ[id_Integer?NonNegative]
:Arguments:      { id }
:ArgumentTypes:  { Integer }
:ReturnType:     Integer
:End:


:Begin:
:Function:       serializeScene
:Pattern:        serializeScene[id_Integer?NonNegative, fn_String]
:Arguments:      { id, fn }
:ArgumentTypes:  { Integer, String }
:ReturnType:     Manual
:End:


:Begin:
:Function:       deserializeScene
:Pattern:        deserializeScene[id_Integer?NonNegative, fn_String]
:Arguments:      { id, fn }
:ArgumentTypes:  { Integer, String }
:ReturnType:     Manual
:End:


:Begin:
:Function:       countVoxelBlocks
:Pattern:        countVoxelBlocks[id_Integer?NonNegative]
:Arguments:      { id }
:ArgumentTypes:  { Integer }
:ReturnType:     Integer
:End:


:Begin:
:Function:       getVoxelBlock
:Pattern:        getVoxelBlock[id_Integer?NonNegative, i_Integer?Positive]
:Arguments:      { id, i }
:ArgumentTypes:  { Integer, Integer }
:ReturnType:     Manual
:End:

:Begin:
:Function:       putVoxelBlock
:Pattern:        putVoxelBlock[
    id_Integer?NonNegative
    (* Manual *)
    , voxelBlockData : { {_,_,_} (*pos*), {__List} (*8^3 voxels' data*) }]
:Arguments:      { id, voxelBlockData }
:ArgumentTypes:  { Integer, Manual }
:ReturnType:     Manual
:End:

:Begin:
:Function:       meshScene
:Pattern:        meshScene[id_Integer?NonNegative, fn_String]
:Arguments:      { id, fn }
:ArgumentTypes:  { Integer, String }
:ReturnType:     Manual
:End:


:Begin:
:Function:       meshSceneWithShader
:Pattern:        meshSceneWithShader[id_Integer?NonNegative, fn_String, shader_String, shaderParam_Real]
:Arguments:      { id, fn, shader, shaderParam }
:ArgumentTypes:  { Integer, String, String, Real }
:ReturnType:     Manual
:End:


:Begin:
:Function:       processFrame
:Pattern:        processFrame[
    id_Integer?NonNegative
    (* Manual *)
    , rgbaByteImage_ /;TensorQ[rgbaByteImage, IntegerQ] && Last@Dimensions@rgbaByteImage == 4
    , depthData_?NumericMatrixQ
    , poseWorldToView_?PoseMatrixQ
    , intrinsicsRgb : NamelessIntrinsicsPattern[]
    , intrinsicsD : NamelessIntrinsicsPattern[]
    , rgbToDepth_?PoseMatrixQ
    ]
:Arguments:      { doTracking, id, rgbaByteImage, depthData, poseWorldToView, intrinsicsRgb, intrinsicsD, rgbToDepth }
:ArgumentTypes:  { Integer, Integer, Manual }
:ReturnType:     Manual
:End:



:Evaluate: Protect@"InfiniTAM2`Private`*"
:Evaluate: End[] 
:Evaluate: $ContextPath = $oldContextPath
