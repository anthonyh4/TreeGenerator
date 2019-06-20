import bpy
import random


















if 'tree' in bpy.data.objects.keys():
    bpy.data.objects['tree'].select = True
    bpy.ops.object.delete()

tree_parameters = {seed:}

TreeSeed = random.randint(1000,5000)

bpy.ops.curve.tree_add(do_update=True, bevel=True, prune=False, showLeaves=False, useArm=False, seed=TreeSeed, handleType='1', levels=3, length=(0.75, 0.5, 1.5, 0.1), lengthV=(0, 0.1, 0, 0), taperCrown=0, branches=(0, 35, 15, 1), curveRes=(8, 16, 8, 1), curve=(0, 20, -40, 0), curveV=(150, 120, 0, 0), curveBack=(0, 20, 0, 0), baseSplits=2, segSplits=(0.1, 0.2, 0.2, 0), splitByLen=True, rMode='rotate', splitAngle=(12, 30, 16, 0), splitAngleV=(0, 10, 20, 0), scale=15, scaleV=5, attractUp=(0, 0, -2.75, -3), attractOut=(0, 0, 0, 0), shape='4', shapeS='4', customShape=(0.5, 1, 0.3, 0.5), branchDist=1.5, nrings=0, baseSize=0.2, baseSize_s=0.25, splitHeight=0.2, splitBias=0, ratio=0.025, minRadius=0.0015, closeTip=False, rootFlare=1, autoTaper=True, taper=(1, 1, 1, 1), radiusTweak=(1, 1, 1, 1), ratioPower=1.75, downAngle=(0, 20, 30, 20), downAngleV=(0, 20, 10, 10), useOldDownAngle=False, useParentAngle=True, rotate=(99.5, 137.5, -60, 140), rotateV=(15, 15, 45, 0), scale0=1, scaleV0=0, pruneWidth=0.5, pruneBase=0.07, pruneWidthPeak=0.6, prunePowerHigh=0.2, prunePowerLow=0.001, pruneRatio=0.8, leaves=150, leafDownAngle=30, leafDownAngleV=10, leafRotate=137.5, leafRotateV=30, leafScale=0.25, leafScaleX=0.2, leafScaleT=0, leafScaleV=0, leafShape='hex', bend=0, leafangle=0, horzLeaves=False, leafDist='10', bevelRes=1, resU=4, armAnim=False, previewArm=False, leafAnim=False, frameRate=1, loopFrames=0, wind=1, gust=1, gustF=0.075, af1=1, af2=1, af3=4, makeMesh=False, armLevels=2, boneStep=(1, 1, 1, 1))
