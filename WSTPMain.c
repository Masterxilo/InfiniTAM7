/*
 * This file automatically produced by wsprep from:
 *	WSTPTemplateFile.tm
 * mprep Revision 18 Copyright (c) Wolfram Research, Inc. 1990-2013
 */

#define MPREP_REVISION 18


#include "wstp.h"

int WSAbort = 0;
int WSDone  = 0;
long WSSpecialCharacter = '\0';
HANDLE WSInstance = (HANDLE)0;
HWND WSIconWindow = (HWND)0;

WSLINK stdlink = 0;
WSEnvironment stdenv = 0;
#if WSINTERFACE >= 3
WSYieldFunctionObject stdyielder = (WSYieldFunctionObject)0;
WSMessageHandlerObject stdhandler = (WSMessageHandlerObject)0;
#else
WSYieldFunctionObject stdyielder = 0;
WSMessageHandlerObject stdhandler = 0;
#endif /* WSINTERFACE >= 3 */

#include <windows.h>

#if defined(__GNUC__)

#	ifdef TCHAR
#		undef TCHAR
#	endif
#	define TCHAR char

#	ifdef PTCHAR
#		undef PTCHAR
#	endif
#	define PTCHAR char *

#	ifdef __TEXT
#		undef __TEXT
#	endif
#	define __TEXT(arg) arg

#	ifdef _tcsrchr
#		undef _tcsrchr
#	endif
#	define _tcsrchr strrchr

#	ifdef _tcscat
#		undef _tcscat
#	endif
#	define _tcscat strcat

#	ifdef _tcsncpy
#		undef _tcsncpy
#	endif
#	define _tcsncpy _fstrncpy
#else
#	include <tchar.h>
#endif

#include <stdlib.h>
#include <string.h>
#if (WIN32_WSTP || WIN64_WSTP || __GNUC__) && !defined(_fstrncpy)
#       define _fstrncpy strncpy
#endif

#ifndef CALLBACK
#define CALLBACK FAR PASCAL
typedef LONG LRESULT;
typedef unsigned int UINT;
typedef WORD WPARAM;
typedef DWORD LPARAM;
#endif


LRESULT CALLBACK WSEXPORT
IconProcedure( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT CALLBACK WSEXPORT
IconProcedure( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch( msg){
	case WM_CLOSE:
		WSDone = 1;
		WSAbort = 1;
		break;
	case WM_QUERYOPEN:
		return 0;
	}
	return DefWindowProc( hWnd, msg, wParam, lParam);
}


#ifdef _UNICODE
#define _APISTR(i) L ## #i
#else
#define _APISTR(i) #i
#endif

#define APISTR(i) _APISTR(i)

HWND WSInitializeIcon( HINSTANCE hInstance, int nCmdShow)
{
	TCHAR path_name[260];
	PTCHAR icon_name;

	WNDCLASS  wc;
	HMODULE hdll;

	WSInstance = hInstance;
	if( ! nCmdShow) return (HWND)0;

	hdll = GetModuleHandle( __TEXT("ml32i" APISTR(WSINTERFACE)));

	(void)GetModuleFileName( hInstance, path_name, 260);

	icon_name = _tcsrchr( path_name, '\\') + 1;
	*_tcsrchr( icon_name, '.') = '\0';


	wc.style = 0;
	wc.lpfnWndProc = IconProcedure;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInstance;

	if( hdll)
		wc.hIcon = LoadIcon( hdll, __TEXT("MLIcon"));
	else
		wc.hIcon = LoadIcon( NULL, IDI_APPLICATION);

	wc.hCursor = LoadCursor( NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)( COLOR_WINDOW + 1);
	wc.lpszMenuName =  NULL;
	wc.lpszClassName = __TEXT("mprepIcon");
	(void)RegisterClass( &wc);

	WSIconWindow = CreateWindow( __TEXT("mprepIcon"), icon_name,
			WS_OVERLAPPEDWINDOW | WS_MINIMIZE, CW_USEDEFAULT,
			CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
			(HWND)0, (HMENU)0, hInstance, (void FAR*)0);

	if( WSIconWindow){
		ShowWindow( WSIconWindow, SW_MINIMIZE);
		UpdateWindow( WSIconWindow);
	}
	return WSIconWindow;
}


#if __BORLANDC__
#pragma argsused
#endif

#if WSINTERFACE >= 3
WSYDEFN( int, WSDefaultYielder, ( WSLINK mlp, WSYieldParameters yp))
#else
WSYDEFN( devyield_result, WSDefaultYielder, ( WSLINK mlp, WSYieldParameters yp))
#endif /* WSINTERFACE >= 3 */
{
	MSG msg;

#if !__BORLANDC__
	mlp = mlp; /* suppress unused warning */
	yp = yp; /* suppress unused warning */
#endif

	if( PeekMessage( &msg, (HWND)0, 0, 0, PM_REMOVE)){
		TranslateMessage( &msg);
		DispatchMessage( &msg);
	}
	return WSDone;
}


/********************************* end header *********************************/


int Get42 P(( void));

#if WSPROTOTYPES
static int _tr0( WSLINK mlp)
#else
static int _tr0(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _rp0;
	if ( ! WSNewPacket(mlp) ) goto L0;

	_rp0 = Get42();

	res = WSAbort ?
		WSPutFunction( mlp, "Abort", 0) : WSPutInteger( mlp, _rp0);

L0:	return res;
} /* _tr0 */


int RunTestsM P(( void));

#if WSPROTOTYPES
static int _tr1( WSLINK mlp)
#else
static int _tr1(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _rp0;
	if ( ! WSNewPacket(mlp) ) goto L0;

	_rp0 = RunTestsM();

	res = WSAbort ?
		WSPutFunction( mlp, "Abort", 0) : WSPutInteger( mlp, _rp0);

L0:	return res;
} /* _tr1 */


int createScene P(( double _tp1));

#if WSPROTOTYPES
static int _tr2( WSLINK mlp)
#else
static int _tr2(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	double _tp1;
	int _rp0;
	if ( ! WSGetReal( mlp, &_tp1) ) goto L0;
	if ( ! WSNewPacket(mlp) ) goto L1;

	_rp0 = createScene(_tp1);

	res = WSAbort ?
		WSPutFunction( mlp, "Abort", 0) : WSPutInteger( mlp, _rp0);
L1: 
L0:	return res;
} /* _tr2 */


double getSceneVoxelSize P(( int _tp1));

#if WSPROTOTYPES
static int _tr3( WSLINK mlp)
#else
static int _tr3(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	double _rp0;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSNewPacket(mlp) ) goto L1;

	_rp0 = getSceneVoxelSize(_tp1);

	res = WSAbort ?
		WSPutFunction( mlp, "Abort", 0) : WSPutReal( mlp, _rp0);
L1: 
L0:	return res;
} /* _tr3 */


int sceneExistsQ P(( int _tp1));

#if WSPROTOTYPES
static int _tr4( WSLINK mlp)
#else
static int _tr4(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	int _rp0;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSNewPacket(mlp) ) goto L1;

	_rp0 = sceneExistsQ(_tp1);

	res = WSAbort ?
		WSPutFunction( mlp, "Abort", 0) : WSPutInteger( mlp, _rp0);
L1: 
L0:	return res;
} /* _tr4 */


void serializeScene P(( int _tp1, const char * _tp2));

#if WSPROTOTYPES
static int _tr5( WSLINK mlp)
#else
static int _tr5(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	const char * _tp2;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSGetString( mlp, &_tp2) ) goto L1;
	if ( ! WSNewPacket(mlp) ) goto L2;

	serializeScene(_tp1, _tp2);

	res = 1;
L2:	WSReleaseString(mlp, _tp2);
L1: 
L0:	return res;
} /* _tr5 */


void deserializeScene P(( int _tp1, const char * _tp2));

#if WSPROTOTYPES
static int _tr6( WSLINK mlp)
#else
static int _tr6(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	const char * _tp2;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSGetString( mlp, &_tp2) ) goto L1;
	if ( ! WSNewPacket(mlp) ) goto L2;

	deserializeScene(_tp1, _tp2);

	res = 1;
L2:	WSReleaseString(mlp, _tp2);
L1: 
L0:	return res;
} /* _tr6 */


int countVoxelBlocks P(( int _tp1));

#if WSPROTOTYPES
static int _tr7( WSLINK mlp)
#else
static int _tr7(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	int _rp0;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSNewPacket(mlp) ) goto L1;

	_rp0 = countVoxelBlocks(_tp1);

	res = WSAbort ?
		WSPutFunction( mlp, "Abort", 0) : WSPutInteger( mlp, _rp0);
L1: 
L0:	return res;
} /* _tr7 */


void getVoxelBlock P(( int _tp1, int _tp2));

#if WSPROTOTYPES
static int _tr8( WSLINK mlp)
#else
static int _tr8(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	int _tp2;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSGetInteger( mlp, &_tp2) ) goto L1;
	if ( ! WSNewPacket(mlp) ) goto L2;

	getVoxelBlock(_tp1, _tp2);

	res = 1;
L2: L1: 
L0:	return res;
} /* _tr8 */


void putVoxelBlock P(( int _tp1));

#if WSPROTOTYPES
static int _tr9( WSLINK mlp)
#else
static int _tr9(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;

	putVoxelBlock(_tp1);

	res = 1;
 
L0:	return res;
} /* _tr9 */


void meshScene P(( int _tp1, const char * _tp2));

#if WSPROTOTYPES
static int _tr10( WSLINK mlp)
#else
static int _tr10(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	const char * _tp2;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSGetString( mlp, &_tp2) ) goto L1;
	if ( ! WSNewPacket(mlp) ) goto L2;

	meshScene(_tp1, _tp2);

	res = 1;
L2:	WSReleaseString(mlp, _tp2);
L1: 
L0:	return res;
} /* _tr10 */


void meshSceneWithShader P(( int _tp1, const char * _tp2, const char * _tp3, double _tp4));

#if WSPROTOTYPES
static int _tr11( WSLINK mlp)
#else
static int _tr11(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	const char * _tp2;
	const char * _tp3;
	double _tp4;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSGetString( mlp, &_tp2) ) goto L1;
	if ( ! WSGetString( mlp, &_tp3) ) goto L2;
	if ( ! WSGetReal( mlp, &_tp4) ) goto L3;
	if ( ! WSNewPacket(mlp) ) goto L4;

	meshSceneWithShader(_tp1, _tp2, _tp3, _tp4);

	res = 1;
L4: L3:	WSReleaseString(mlp, _tp3);
L2:	WSReleaseString(mlp, _tp2);
L1: 
L0:	return res;
} /* _tr11 */


void processFrame P(( int _tp1));

#if WSPROTOTYPES
static int _tr12( WSLINK mlp)
#else
static int _tr12(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;

	processFrame(_tp1);

	res = 1;
 
L0:	return res;
} /* _tr12 */


void initAD P(( int _tp1));

#if WSPROTOTYPES
static int _tr13( WSLINK mlp)
#else
static int _tr13(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSNewPacket(mlp) ) goto L1;

	initAD(_tp1);

	res = 1;
L1: 
L0:	return res;
} /* _tr13 */


void initFineADFromCoarseAD P(( int _tp1, int _tp2));

#if WSPROTOTYPES
static int _tr14( WSLINK mlp)
#else
static int _tr14(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	int _tp2;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSGetInteger( mlp, &_tp2) ) goto L1;
	if ( ! WSNewPacket(mlp) ) goto L2;

	initFineADFromCoarseAD(_tp1, _tp2);

	res = 1;
L2: L1: 
L0:	return res;
} /* _tr14 */


void refineScene P(( int _tp1, double _tp2, double _tp3, double _tp4, double _tp5, double * _tp6, long _tpl6));

#if WSPROTOTYPES
static int _tr15( WSLINK mlp)
#else
static int _tr15(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	double _tp2;
	double _tp3;
	double _tp4;
	double _tp5;
	double * _tp6;
	long _tpl6;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSGetReal( mlp, &_tp2) ) goto L1;
	if ( ! WSGetReal( mlp, &_tp3) ) goto L2;
	if ( ! WSGetReal( mlp, &_tp4) ) goto L3;
	if ( ! WSGetReal( mlp, &_tp5) ) goto L4;
	if ( ! WSGetRealList( mlp, &_tp6, &_tpl6) ) goto L5;
	if ( ! WSNewPacket(mlp) ) goto L6;

	refineScene(_tp1, _tp2, _tp3, _tp4, _tp5, _tp6, _tpl6);

	res = 1;
L6:	WSReleaseReal64List(mlp, _tp6, _tpl6);
L5: L4: L3: L2: L1: 
L0:	return res;
} /* _tr15 */


double sceneEnergy P(( int _tp1, double _tp2, double _tp3, double _tp4, double _tp5, double * _tp6, long _tpl6));

#if WSPROTOTYPES
static int _tr16( WSLINK mlp)
#else
static int _tr16(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	double _tp2;
	double _tp3;
	double _tp4;
	double _tp5;
	double * _tp6;
	long _tpl6;
	double _rp0;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSGetReal( mlp, &_tp2) ) goto L1;
	if ( ! WSGetReal( mlp, &_tp3) ) goto L2;
	if ( ! WSGetReal( mlp, &_tp4) ) goto L3;
	if ( ! WSGetReal( mlp, &_tp5) ) goto L4;
	if ( ! WSGetRealList( mlp, &_tp6, &_tpl6) ) goto L5;
	if ( ! WSNewPacket(mlp) ) goto L6;

	_rp0 = sceneEnergy(_tp1, _tp2, _tp3, _tp4, _tp5, _tp6, _tpl6);

	res = WSAbort ?
		WSPutFunction( mlp, "Abort", 0) : WSPutReal( mlp, _rp0);
L6:	WSReleaseReal64List(mlp, _tp6, _tpl6);
L5: L4: L3: L2: L1: 
L0:	return res;
} /* _tr16 */


void dumpSceneVoxelPositions P(( int _tp1, const char * _tp2));

#if WSPROTOTYPES
static int _tr17( WSLINK mlp)
#else
static int _tr17(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	const char * _tp2;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSGetString( mlp, &_tp2) ) goto L1;
	if ( ! WSNewPacket(mlp) ) goto L2;

	dumpSceneVoxelPositions(_tp1, _tp2);

	res = 1;
L2:	WSReleaseString(mlp, _tp2);
L1: 
L0:	return res;
} /* _tr17 */


void dumpSceneVoxelPositionsBlockwise P(( int _tp1, const char * _tp2));

#if WSPROTOTYPES
static int _tr18( WSLINK mlp)
#else
static int _tr18(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	const char * _tp2;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSGetString( mlp, &_tp2) ) goto L1;
	if ( ! WSNewPacket(mlp) ) goto L2;

	dumpSceneVoxelPositionsBlockwise(_tp1, _tp2);

	res = 1;
L2:	WSReleaseString(mlp, _tp2);
L1: 
L0:	return res;
} /* _tr18 */


void dumpSceneOptimizationBlocks P(( int _tp1, const char * _tp2, int _tp3, int _tp4, int _tp5));

#if WSPROTOTYPES
static int _tr19( WSLINK mlp)
#else
static int _tr19(mlp) WSLINK mlp;
#endif
{
	int	res = 0;
	int _tp1;
	const char * _tp2;
	int _tp3;
	int _tp4;
	int _tp5;
	if ( ! WSGetInteger( mlp, &_tp1) ) goto L0;
	if ( ! WSGetString( mlp, &_tp2) ) goto L1;
	if ( ! WSGetInteger( mlp, &_tp3) ) goto L2;
	if ( ! WSGetInteger( mlp, &_tp4) ) goto L3;
	if ( ! WSGetInteger( mlp, &_tp5) ) goto L4;
	if ( ! WSNewPacket(mlp) ) goto L5;

	dumpSceneOptimizationBlocks(_tp1, _tp2, _tp3, _tp4, _tp5);

	res = 1;
L5: L4: L3: L2:	WSReleaseString(mlp, _tp2);
L1: 
L0:	return res;
} /* _tr19 */


static struct func {
	int   f_nargs;
	int   manual;
	int   (*f_func)P((WSLINK));
	const char  *f_name;
	} _tramps[20] = {
		{ 0, 0, _tr0, "Get42" },
		{ 0, 0, _tr1, "RunTestsM" },
		{ 1, 0, _tr2, "createScene" },
		{ 1, 0, _tr3, "getSceneVoxelSize" },
		{ 1, 0, _tr4, "sceneExistsQ" },
		{ 2, 0, _tr5, "serializeScene" },
		{ 2, 0, _tr6, "deserializeScene" },
		{ 1, 0, _tr7, "countVoxelBlocks" },
		{ 2, 0, _tr8, "getVoxelBlock" },
		{ 1, 2, _tr9, "putVoxelBlock" },
		{ 2, 0, _tr10, "meshScene" },
		{ 4, 0, _tr11, "meshSceneWithShader" },
		{ 1, 2, _tr12, "processFrame" },
		{ 1, 0, _tr13, "initAD" },
		{ 2, 0, _tr14, "initFineADFromCoarseAD" },
		{ 6, 0, _tr15, "refineScene" },
		{ 6, 0, _tr16, "sceneEnergy" },
		{ 2, 0, _tr17, "dumpSceneVoxelPositions" },
		{ 2, 0, _tr18, "dumpSceneVoxelPositionsBlockwise" },
		{ 5, 0, _tr19, "dumpSceneOptimizationBlocks" }
		};

static const char* evalstrs[] = {
	"$oldContextPath = $ContextPath; $ContextPath = {\"System`\", \"Glob",
	"al`\"}; (* these are the only dependencies *)",
	(const char*)0,
	"Begin@\"InfiniTAM2`Private`\" (* create everythin in InfiniTAM2`Pr",
	"ivate`* *)",
	(const char*)0,
	"UnprotectClearAll@\"InfiniTAM2`Private`*\" (* create everythin in ",
	"InfiniTAM`Private`* *)",
	(const char*)0,
	"Protect@\"InfiniTAM2`Private`*\"",
	(const char*)0,
	"End[]",
	(const char*)0,
	"$ContextPath = $oldContextPath",
	(const char*)0,
	(const char*)0
};
#define CARDOF_EVALSTRS 6

static int _definepattern P(( WSLINK, char*, char*, int));

static int _doevalstr P(( WSLINK, int));

int  _WSDoCallPacket P(( WSLINK, struct func[], int));


#if WSPROTOTYPES
int WSInstall( WSLINK mlp)
#else
int WSInstall(mlp) WSLINK mlp;
#endif
{
	int _res;
	_res = WSConnect(mlp);
	if (_res) _res = _definepattern(mlp, (char *)"Get42[]", (char *)"{  }", 0);
	if (_res) _res = _definepattern(mlp, (char *)"RunTestsM[]", (char *)"{  }", 1);
	if (_res) _res = _doevalstr( mlp, 0);
	if (_res) _res = _doevalstr( mlp, 1);
	if (_res) _res = _doevalstr( mlp, 2);
	if (_res) _res = _definepattern(mlp, (char *)"createScene[voxelSize_Real]", (char *)"{ voxelSize }", 2);
	if (_res) _res = _definepattern(mlp, (char *)"getSceneVoxelSize[id_Integer?NonNegative]", (char *)"{ id }", 3);
	if (_res) _res = _definepattern(mlp, (char *)"sceneExistsQ[id_Integer?NonNegative]", (char *)"{ id }", 4);
	if (_res) _res = _definepattern(mlp, (char *)"serializeScene[id_Integer?NonNegative, fn_String]", (char *)"{ id, fn }", 5);
	if (_res) _res = _definepattern(mlp, (char *)"deserializeScene[id_Integer?NonNegative, fn_String]", (char *)"{ id, fn }", 6);
	if (_res) _res = _definepattern(mlp, (char *)"countVoxelBlocks[id_Integer?NonNegative]", (char *)"{ id }", 7);
	if (_res) _res = _definepattern(mlp, (char *)"getVoxelBlock[id_Integer?NonNegative, i_Integer?Positive]", (char *)"{ id, i }", 8);
	if (_res) _res = _definepattern(mlp, (char *)"putVoxelBlock[     id_Integer?NonNegative     (* Manual *)     , voxelBlockData : { {_,_,_} (*pos*), {__List} (*8^3 voxels' data*) }]", (char *)"{ id, voxelBlockData }", 9);
	if (_res) _res = _definepattern(mlp, (char *)"meshScene[id_Integer?NonNegative, fn_String]", (char *)"{ id, fn }", 10);
	if (_res) _res = _definepattern(mlp, (char *)"meshSceneWithShader[id_Integer?NonNegative, fn_String, shader_String, shaderParam_Real]", (char *)"{ id, fn, shader, shaderParam }", 11);
	if (_res) _res = _definepattern(mlp, (char *)"processFrame[     id_Integer?NonNegative     (* Manual *)     , rgbaByteImage_ /;TensorQ[rgbaByteImage, IntegerQ] && Last@Dimensions@rgbaByteImage == 4     , depthData_?NumericMatrixQ     , poseWorldToView_?PoseMatrixQ     , intrinsicsRgb : NamelessIntrinsicsPattern[]     , intrinsicsD : NamelessIntrinsicsPattern[]     , rgbToDepth_?PoseMatrixQ     ]", (char *)"{ id, rgbaByteImage, depthData, poseWorldToView, intrinsicsRgb, intrinsicsD, rgbToDepth }", 12);
	if (_res) _res = _definepattern(mlp, (char *)"initAD[id_Integer?NonNegative]", (char *)"{ id }", 13);
	if (_res) _res = _definepattern(mlp, (char *)"initFineADFromCoarseAD[fineid_Integer?NonNegative, coarseid_Integer?NonNegative] /; fineid != coarseid", (char *)"{ fineid, coarseid }", 14);
	if (_res) _res = _definepattern(mlp, (char *)"refineScene[id_Integer, eg_Real, er_Real, es_Real, ea_Real, l : {__Real}]", (char *)"{ id, eg, er, es, ea, l }", 15);
	if (_res) _res = _definepattern(mlp, (char *)"sceneEnergy[id_Integer, eg_Real, er_Real, es_Real, ea_Real, l : {__Real}]", (char *)"{ id, eg, er, es, ea, l }", 16);
	if (_res) _res = _definepattern(mlp, (char *)"dumpSceneVoxelPositions[id_Integer?NonNegative, fn_String]", (char *)"{ id, fn }", 17);
	if (_res) _res = _definepattern(mlp, (char *)"dumpSceneVoxelPositionsBlockwise[id_Integer?NonNegative, fn_String]", (char *)"{ id, fn }", 18);
	if (_res) _res = _definepattern(mlp, (char *)"dumpSceneOptimizationBlocks[id_Integer?NonNegative, fn_String, {offsetx_Integer,offsety_Integer,offsetz_Integer}]", (char *)"{ id, fn, offsetx, offsety, offsetz }", 19);
	if (_res) _res = _doevalstr( mlp, 3);
	if (_res) _res = _doevalstr( mlp, 4);
	if (_res) _res = _doevalstr( mlp, 5);
	if (_res) _res = WSPutSymbol( mlp, "End");
	if (_res) _res = WSFlush( mlp);
	return _res;
} /* WSInstall */


#if WSPROTOTYPES
int WSDoCallPacket( WSLINK mlp)
#else
int WSDoCallPacket( mlp) WSLINK mlp;
#endif
{
	return _WSDoCallPacket( mlp, _tramps, 20);
} /* WSDoCallPacket */

/******************************* begin trailer ********************************/

#ifndef EVALSTRS_AS_BYTESTRINGS
#	define EVALSTRS_AS_BYTESTRINGS 1
#endif

#if CARDOF_EVALSTRS
static int  _doevalstr( WSLINK mlp, int n)
{
	long bytesleft, charsleft, bytesnow;
#if !EVALSTRS_AS_BYTESTRINGS
	long charsnow;
#endif
	char **s, **p;
	char *t;

	s = (char **)evalstrs;
	while( n-- > 0){
		if( *s == 0) break;
		while( *s++ != 0){}
	}
	if( *s == 0) return 0;
	bytesleft = 0;
	charsleft = 0;
	p = s;
	while( *p){
		t = *p; while( *t) ++t;
		bytesnow = (long)(t - *p);
		bytesleft += bytesnow;
		charsleft += bytesnow;
#if !EVALSTRS_AS_BYTESTRINGS
		t = *p;
		charsleft -= WSCharacterOffset( &t, t + bytesnow, bytesnow);
		/* assert( t == *p + bytesnow); */
#endif
		++p;
	}


	WSPutNext( mlp, WSTKSTR);
#if EVALSTRS_AS_BYTESTRINGS
	p = s;
	while( *p){
		t = *p; while( *t) ++t;
		bytesnow = (long)(t - *p);
		bytesleft -= bytesnow;
		WSPut8BitCharacters( mlp, bytesleft, (unsigned char*)*p, bytesnow);
		++p;
	}
#else
	WSPut7BitCount( mlp, (long_st)charsleft, (long_st)bytesleft);

	p = s;
	while( *p){
		t = *p; while( *t) ++t;
		bytesnow = t - *p;
		bytesleft -= bytesnow;
		t = *p;
		charsnow = bytesnow - WSCharacterOffset( &t, t + bytesnow, bytesnow);
		/* assert( t == *p + bytesnow); */
		charsleft -= charsnow;
		WSPut7BitCharacters(  mlp, charsleft, *p, bytesnow, charsnow);
		++p;
	}
#endif
	return WSError( mlp) == WSEOK;
}
#endif /* CARDOF_EVALSTRS */


static int  _definepattern( WSLINK mlp, char *patt, char *args, int func_n)
{
	WSPutFunction( mlp, "DefineExternal", (long)3);
	  WSPutString( mlp, patt);
	  WSPutString( mlp, args);
	  WSPutInteger( mlp, func_n);
	return !WSError(mlp);
} /* _definepattern */


int _WSDoCallPacket( WSLINK mlp, struct func functable[], int nfuncs)
{
#if WSINTERFACE >= 4
	int len;
#else
	long len;
#endif
	int n, res = 0;
	struct func* funcp;

	if( ! WSGetInteger( mlp, &n) ||  n < 0 ||  n >= nfuncs) goto L0;
	funcp = &functable[n];

	if( funcp->f_nargs >= 0
#if WSINTERFACE >= 4
	&& ( ! WSTestHead(mlp, "List", &len)
#else
	&& ( ! WSCheckFunction(mlp, "List", &len)
#endif
	     || ( !funcp->manual && (len != funcp->f_nargs))
	     || (  funcp->manual && (len <  funcp->f_nargs))
	   )
	) goto L0;

	stdlink = mlp;
	res = (*funcp->f_func)( mlp);

L0:	if( res == 0)
		res = WSClearError( mlp) && WSPutSymbol( mlp, "$Failed");
	return res && WSEndPacket( mlp) && WSNewPacket( mlp);
} /* _WSDoCallPacket */


wsapi_packet WSAnswer( WSLINK mlp)
{
	wsapi_packet pkt = 0;
#if WSINTERFACE >= 4
	int waitResult;

	while( ! WSDone && ! WSError(mlp)
		&& (waitResult = WSWaitForLinkActivity(mlp),waitResult) &&
		waitResult == WSWAITSUCCESS && (pkt = WSNextPacket(mlp), pkt) &&
		pkt == CALLPKT)
	{
		WSAbort = 0;
		if(! WSDoCallPacket(mlp))
			pkt = 0;
	}
#else
	while( !WSDone && !WSError(mlp) && (pkt = WSNextPacket(mlp), pkt) && pkt == CALLPKT){
		WSAbort = 0;
		if( !WSDoCallPacket(mlp)) pkt = 0;
	}
#endif
	WSAbort = 0;
	return pkt;
}



/*
	Module[ { me = $ParentLink},
		$ParentLink = contents of RESUMEPKT;
		Message[ MessageName[$ParentLink, "notfe"], me];
		me]
*/

static int refuse_to_be_a_frontend( WSLINK mlp)
{
	int pkt;

	WSPutFunction( mlp, "EvaluatePacket", 1);
	  WSPutFunction( mlp, "Module", 2);
	    WSPutFunction( mlp, "List", 1);
		  WSPutFunction( mlp, "Set", 2);
		    WSPutSymbol( mlp, "me");
	        WSPutSymbol( mlp, "$ParentLink");
	  WSPutFunction( mlp, "CompoundExpression", 3);
	    WSPutFunction( mlp, "Set", 2);
	      WSPutSymbol( mlp, "$ParentLink");
	      WSTransferExpression( mlp, mlp);
	    WSPutFunction( mlp, "Message", 2);
	      WSPutFunction( mlp, "MessageName", 2);
	        WSPutSymbol( mlp, "$ParentLink");
	        WSPutString( mlp, "notfe");
	      WSPutSymbol( mlp, "me");
	    WSPutSymbol( mlp, "me");
	WSEndPacket( mlp);

	while( (pkt = WSNextPacket( mlp), pkt) && pkt != SUSPENDPKT)
		WSNewPacket( mlp);
	WSNewPacket( mlp);
	return WSError( mlp) == WSEOK;
}


#if WSINTERFACE >= 3
int WSEvaluate( WSLINK mlp, char *s)
#else
int WSEvaluate( WSLINK mlp, charp_ct s)
#endif /* WSINTERFACE >= 3 */
{
	if( WSAbort) return 0;
	return WSPutFunction( mlp, "EvaluatePacket", 1L)
		&& WSPutFunction( mlp, "ToExpression", 1L)
		&& WSPutString( mlp, s)
		&& WSEndPacket( mlp);
}


#if WSINTERFACE >= 3
int WSEvaluateString( WSLINK mlp, char *s)
#else
int WSEvaluateString( WSLINK mlp, charp_ct s)
#endif /* WSINTERFACE >= 3 */
{
	int pkt;
	if( WSAbort) return 0;
	if( WSEvaluate( mlp, s)){
		while( (pkt = WSAnswer( mlp), pkt) && pkt != RETURNPKT)
			WSNewPacket( mlp);
		WSNewPacket( mlp);
	}
	return WSError( mlp) == WSEOK;
} /* WSEvaluateString */


#if __BORLANDC__
#pragma argsused
#endif

#if WSINTERFACE >= 3
WSMDEFN( void, WSDefaultHandler, ( WSLINK mlp, int message, int n))
#else
WSMDEFN( void, WSDefaultHandler, ( WSLINK mlp, unsigned long message, unsigned long n))
#endif /* WSINTERFACE >= 3 */
{
#if !__BORLANDC__
	mlp = (WSLINK)0; /* suppress unused warning */
	n = 0;          /* suppress unused warning */
#endif

	switch (message){
	case WSTerminateMessage:
		WSDone = 1;
	case WSInterruptMessage:
	case WSAbortMessage:
		WSAbort = 1;
	default:
		return;
	}
}



#if WSINTERFACE >= 3
static int _WSMain( char **argv, char **argv_end, char *commandline)
#else
static int _WSMain( charpp_ct argv, charpp_ct argv_end, charp_ct commandline)
#endif /* WSINTERFACE >= 3 */
{
	WSLINK mlp;
#if WSINTERFACE >= 3
	int err;
#else
	long err;
#endif /* WSINTERFACE >= 3 */

#if WSINTERFACE >= 4
	if( !stdenv)
		stdenv = WSInitialize( (WSEnvironmentParameter)0);
#else
	if( !stdenv)
		stdenv = WSInitialize( (WSParametersPointer)0);
#endif

	if( stdenv == (WSEnvironment)0) goto R0;

	if( !stdyielder)
#if WSINTERFACE >= 3
		stdyielder = (WSYieldFunctionObject)WSDefaultYielder;
#else
		stdyielder = WSCreateYieldFunction( stdenv,
			NewWSYielderProc( WSDefaultYielder), 0);
#endif /* WSINTERFACE >= 3 */


#if WSINTERFACE >= 3
	if( !stdhandler)
		stdhandler = (WSMessageHandlerObject)WSDefaultHandler;
#else
	if( !stdhandler)
		stdhandler = WSCreateMessageHandler( stdenv,
			NewWSHandlerProc( WSDefaultHandler), 0);
#endif /* WSINTERFACE >= 3 */


	mlp = commandline
		? WSOpenString( stdenv, commandline, &err)
#if WSINTERFACE >= 3
		: WSOpenArgcArgv( stdenv, (int)(argv_end - argv), argv, &err);
#else
		: WSOpenArgv( stdenv, argv, argv_end, &err);
#endif
	if( mlp == (WSLINK)0){
		WSAlert( stdenv, WSErrorString( stdenv, err));
		goto R1;
	}

	if( WSIconWindow){
#define TEXTBUFLEN 64
		TCHAR textbuf[TEXTBUFLEN];
		PTCHAR tmlname;
		const char *mlname;
		size_t namelen, i;
		int len;
		len = GetWindowText(WSIconWindow, textbuf, 62 );
		mlname = WSName(mlp);
		namelen = strlen(mlname);
		tmlname = (PTCHAR)malloc((namelen + 1)*sizeof(TCHAR));
		if(tmlname == NULL) goto R2;

		for(i = 0; i < namelen; i++){
			tmlname[i] = mlname[i];
		}
		tmlname[namelen] = '\0';
		
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
		_tcscat_s( textbuf + len, TEXTBUFLEN - len, __TEXT("("));
		_tcsncpy_s(textbuf + len + 1, TEXTBUFLEN - len - 1, tmlname, TEXTBUFLEN - len - 3);
		textbuf[TEXTBUFLEN - 2] = '\0';
		_tcscat_s(textbuf, TEXTBUFLEN, __TEXT(")"));
#else
		_tcscat( textbuf + len, __TEXT("("));
		_tcsncpy( textbuf + len + 1, tmlname, TEXTBUFLEN - len - 3);
		textbuf[TEXTBUFLEN - 2] = '\0';
		_tcscat( textbuf, __TEXT(")"));
#endif
		textbuf[len + namelen + 2] = '\0';
		free(tmlname);
		SetWindowText( WSIconWindow, textbuf);
	}

	if( WSInstance){
		if( stdyielder) WSSetYieldFunction( mlp, stdyielder);
		if( stdhandler) WSSetMessageHandler( mlp, stdhandler);
	}

	if( WSInstall( mlp))
		while( WSAnswer( mlp) == RESUMEPKT){
			if( ! refuse_to_be_a_frontend( mlp)) break;
		}

R2:	WSClose( mlp);
R1:	WSDeinitialize( stdenv);
	stdenv = (WSEnvironment)0;
R0:	return !WSDone;
} /* _WSMain */


#if WSINTERFACE >= 3
int WSMainString( char *commandline)
#else
int WSMainString( charp_ct commandline)
#endif /* WSINTERFACE >= 3 */
{
#if WSINTERFACE >= 3
	return _WSMain( (char **)0, (char **)0, commandline);
#else
	return _WSMain( (charpp_ct)0, (charpp_ct)0, commandline);
#endif /* WSINTERFACE >= 3 */
}

int WSMainArgv( char** argv, char** argv_end) /* note not FAR pointers */
{   
	static char FAR * far_argv[128];
	int count = 0;
	
	while(argv < argv_end)
		far_argv[count++] = *argv++;
		 
#if WSINTERFACE >= 3
	return _WSMain( far_argv, far_argv + count, (char *)0);
#else
	return _WSMain( far_argv, far_argv + count, (charp_ct)0);
#endif /* WSINTERFACE >= 3 */

}

#if WSINTERFACE >= 3
int WSMain( int argc, char **argv)
#else
int WSMain( int argc, charpp_ct argv)
#endif /* WSINTERFACE >= 3 */
{
#if WSINTERFACE >= 3
 	return _WSMain( argv, argv + argc, (char *)0);
#else
 	return _WSMain( argv, argv + argc, (charp_ct)0);
#endif /* WSINTERFACE >= 3 */
}
 
