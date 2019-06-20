import bpy
import random
import json

#### start blender --background --python treeGenerator.py  --> This is the command (windows)  --> need linux (ubuntu)
#### blender --background --python treeGenerator.py &
# bpy.data.objects.remove(bpy.data.objects['Cube'])

# bpy.ops.create
# bpy.ops.wm.open_mainfile(filepath="C:\\Users\\Anthony Hernandez\\Desktop\\Blender Files\\base.blend") ##CHANGE THIS TODO
bpy.context.scene.render.engine = 'CYCLES'

parPaths = 'C:/Users/Anthony Hernandez/Desktop/Julies Python Supervised machine learning/predictions/predict.json'

with open(parPaths) as json_file:
	data = json.load(json_file)

for i in data.keys():
	if 'tree' in bpy.data.objects.keys():
		bpy.data.objects['tree'].select = True
		bpy.ops.object.delete()

	tree_parameters = data[i]

	for j in ['length', 'lengthV', 'segSplits', 'attractOut']:
		tree_parameters[j] = (tree_parameters[j+"1"],tree_parameters[j+'2'],tree_parameters[j+'3'],tree_parameters[j+'4'])
		for x in range(1,5):
			tree_parameters.pop(j+str(x))
	tree_parameters['ratio'] = tree_parameters['ratio']/5
	tree_parameters['baseSize'] = tree_parameters['baseSize']/2
	# print(tree_parameters)
	
	bpy.ops.curve.tree_add(do_update=True, chooseSet='0', bevel=True, prune=False, showLeaves=False, useArm=False, seed=0, 
		handleType='0', levels=2, length=tree_parameters["length"], lengthV=tree_parameters["lengthV"], taperCrown=0, branches=(0, 36, 7, 10), 
		curveRes=(8, 5, 3, 1), curve=(0, -40, -30, 0), curveV=(100, 100, 100, 0), curveBack=(0, 0, 0, 0), baseSplits=0, 
		segSplits=tree_parameters["segSplits"], splitByLen=True, rMode='rotate', splitAngle=(0, 22, 25, 0), splitAngleV=(0, 5, 0, 0), 
		scale=4, scaleV=1, attractUp=(2, 0, 0.5, 0.5), attractOut=tree_parameters["attractOut"], shape='8', shapeS='4', customShape=(0.9, 1, 0.2, 0.2), 
		branchDist=1.6, nrings=7, baseSize=tree_parameters["baseSize"], baseSize_s=tree_parameters["baseSize_s"], splitHeight=0.2, splitBias=0, ratio=tree_parameters["ratio"], minRadius=0.0015, 
		closeTip=False, rootFlare=1, autoTaper=True, taper=(1, 1, 1, 1), radiusTweak=(1, 1, 1, 1), ratioPower=1, downAngle=(90, 110, 45, 45), 
		downAngleV=(0, 42, 10, 10), useOldDownAngle=False, useParentAngle=True, rotate=(99.5, 137.5, -90, 137.5), rotateV=(15, 0, 0, 0), 
		scale0=1, scaleV0=0.1, pruneWidth=tree_parameters["pruneWidth"], pruneBase=0.3, pruneWidthPeak=0.6, prunePowerHigh=0.5, prunePowerLow=0.001, pruneRatio=1, 
		leaves=500, leafDownAngle=65, leafDownAngleV=55, leafRotate=137.5, leafRotateV=30, leafScale=0.2, leafScaleX=0.02, leafScaleT=0.25, 
		leafScaleV=0.1, leafShape='rect', bend=0, leafangle=-10, horzLeaves=False, leafDist='3', bevelRes=2, resU=4, armAnim=False, 
		previewArm=False, leafAnim=False, frameRate=1, loopFrames=0, wind=1, gust=1, gustF=0.075, af1=1, af2=1, af3=4, makeMesh=False, 
		armLevels=2, boneStep=(1, 1, 1, 1))

	bpy.data.objects['tree'].select = True
	
	##bpy.ops.view3d.viewnumpad(type='FRONT', align_active=True)
	##bpy.ops.view3d.camera_to_view()
	bpy.ops.view3d.camera_to_view_selected()
	## ##CHANGE THESE TODO
	bpy.data.scenes['Scene'].render.filepath = 'C:/Users/Anthony Hernandez/Desktop/Blender Files/model results/'+str(i)
	f = open('C:/Users/Anthony Hernandez/Desktop/Blender Files/model results/'+str(i)[:-3]+'txt', "w+")   # 'r' for reading and 'w' for writing
	f.write(str(tree_parameters))    # Write inside file 
	bpy.ops.render.render( write_still=True )
	bpy.data.objects['tree'].select = False

