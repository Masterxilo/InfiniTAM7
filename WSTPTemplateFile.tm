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
:Function:       buildSphereScene
:Pattern:        buildSphereScene[id_Integer?NonNegative, rad_Real]
:Arguments:      { id, rad }
:ArgumentTypes:  { Integer, Real }
:ReturnType:     Manual
:End:

:Begin:
:Function:       buildWallScene
:Pattern:        buildWallScene[id_Integer?NonNegative]
:Arguments:      { id }
:ArgumentTypes:  { Integer }
:ReturnType:     Manual
:End:


:Begin:
:Function:       computeArtificialDirectionalLighting
:Pattern:        computeArtificialDirectionalLighting[id_Integer?NonNegative, dir : {_,_,_}?NumericVectorQ]
:Arguments:      { id, dir}
:ArgumentTypes:  { Integer, RealList }
:ReturnType:     Manual
:End:


:Begin:
:Function:       computeArtificialSphericalHarmonic9Lighting
:Pattern:        computeArtificialSphericalHarmonic9Lighting[id_Integer?NonNegative, l_?NumericVectorQ /; Length@l == 9]
:Arguments:      { id, l }
:ArgumentTypes:  { Integer, RealList }
:ReturnType:     Manual
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
:Arguments:      { id, rgbaByteImage, depthData, poseWorldToView, intrinsicsRgb, intrinsicsD, rgbToDepth }
:ArgumentTypes:  { Integer, Manual }
:ReturnType:     Manual
:End:


:Begin:
:Function:       initAD
:Pattern:        initAD[id_Integer?NonNegative]
:Arguments:      { id }
:ArgumentTypes:  { Integer }
:ReturnType:     Manual
:End:


:Begin:
:Function:       perturbD
:Pattern:        perturbD[id_Integer?NonNegative]
:Arguments:      { id }
:ArgumentTypes:  { Integer }
:ReturnType:     Manual
:End:

:Begin:
:Function:       smoothSDF
:Pattern:        smoothSDF[id_Integer?NonNegative]
:Arguments:      { id }
:ArgumentTypes:  { Integer }
:ReturnType:     Manual
:End:



:Begin:
:Function:       initFineADFromCoarseAD
:Pattern:        initFineADFromCoarseAD[fineid_Integer?NonNegative, coarseid_Integer?NonNegative] /; fineid != coarseid
:Arguments:      { fineid, coarseid }
:ArgumentTypes:  { Integer, Integer }
:ReturnType:     Manual
:End:

:Begin:
:Function:       initCoarseFromFine
:Pattern:        initCoarseFromFine[coarseid_Integer?NonNegative, fineid_Integer?NonNegative] /; fineid != coarseid
:Arguments:      { coarseid, fineid }
:ArgumentTypes:  { Integer, Integer }
:ReturnType:     Manual
:End:

:Begin:
:Function:       refineScene
:Pattern:        refineScene[id_Integer, eg_Real, er_Real, es_Real, ea_Real, l : {__Real}]
:Arguments:      { id, eg, er, es, ea, l }
:ArgumentTypes:  { Integer, Real, Real, Real, Real, RealList }
:ReturnType:     Manual
:End:



:Begin:
:Function:       sceneEnergy
:Pattern:        sceneEnergy[id_Integer, eg_Real, er_Real, es_Real, ea_Real, l : {__Real}]
:Arguments:      { id, eg, er, es, ea, l }
:ArgumentTypes:  { Integer, Real, Real, Real, Real, RealList }
:ReturnType:     Real
:End:


:Begin:
:Function:       dumpSceneVoxelPositions
:Pattern:        dumpSceneVoxelPositions[id_Integer?NonNegative, fn_String]
:Arguments:      { id, fn }
:ArgumentTypes:  { Integer, String }
:ReturnType:     Manual
:End:

:Begin:
:Function:       dumpSceneVoxelPositionsBlockwise
:Pattern:        dumpSceneVoxelPositionsBlockwise[id_Integer?NonNegative, fn_String]
:Arguments:      { id, fn }
:ArgumentTypes:  { Integer, String }
:ReturnType:     Manual
:End:

:Begin:
:Function:       dumpSceneOptimizationBlocks
:Pattern:        dumpSceneOptimizationBlocks[id_Integer?NonNegative, fn_String, {offsetx_Integer,offsety_Integer,offsetz_Integer}]
:Arguments:      { id, fn, offsetx, offsety, offsetz }
:ArgumentTypes:  { Integer, String, Integer, Integer, Integer }
:ReturnType:     Manual
:End:


:Begin:
:Function:       estimateLighting
:Pattern:        estimateLighting[id_Integer?NonNegative]
:Arguments:      { id }
:ArgumentTypes:  { Integer }
:ReturnType:     Manual
:End:

:Evaluate: Protect@"InfiniTAM2`Private`*"
:Evaluate: End[] 
:Evaluate: $ContextPath = $oldContextPath
