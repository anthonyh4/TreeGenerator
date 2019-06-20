import bpy
import random

#### start blender --background --python treeGenerator.py  --> This is the command (windows)  --> need linux (ubuntu)
#### blender --background --python treeGenerator.py &
# bpy.data.objects.remove(bpy.data.objects['Cube'])

# bpy.ops.create
# bpy.ops.wm.open_mainfile(filepath="C:\\Users\\Anthony Hernandez\\Desktop\\Blender Files\\base.blend") ##CHANGE THIS TODO
bpy.context.scene.render.engine = 'CYCLES'

for i in range(21, 301):
	if 'tree' in bpy.data.objects.keys():
		bpy.data.objects['tree'].select = True
		bpy.ops.object.delete()
		# bpy.data.objects['leaves'].select = True
		# bpy.ops.object.delete()


	''' Randomly Sample along the following parameters '''
	# tree_parameters = {
	# 	"bevel": True,
	# 	"showLeaves": False,
	# 	"seed": random.randint(1000,50000),
	# 	"levels": 4,
	# 	"length": (random.uniform(0.3,1.0),random.uniform(0.3,1.0),random.uniform(0.3,1.0),random.uniform(0.3,1.0)),
	# 	"lengthV": (random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3)),
	# 	"branches": (0,random.randint(10,50),random.randint(5,25),random.randint(0,10)),
	# 	"curveRes": (random.randint(5,10),random.randint(15,20),random.randint(5,10),random.randint(1,5)),
	# 	"curve": (random.randint(-5,5),random.randint(-5,5),random.randint(-5,5),0),
	# 	"curveV": (random.randint(0,10),random.randint(0,10),random.randint(0,10),random.randint(0,10)),
	# 	"baseSplits": random.randint(0,2),
	# 	"segSplits": (random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3)),
	# 	"splitByLen": True,
	# 	"splitAngle": (random.randint(2,20),random.randint(10,30),random.randint(2,15),random.randint(1,8)),
	# 	"splitAngleV": (random.randint(1,5),random.randint(7,15),random.randint(1,5),random.randint(1,2)),
	# 	"scale": 20,
	# 	"scaleV": 0,
	# 	"attractUp": (random.uniform(-0.3,0.3),random.uniform(-0.3,0.3),random.uniform(-0.3,0.3),random.uniform(-0.3,0.3)),
	# 	"attractOut": (random.uniform(-0.3,0.3),random.uniform(-0.3,0.3),random.uniform(-0.3,0.3),random.uniform(-0.3,0.3)),
	# 	"branchDist": random.uniform(1.0,2.0),
	# 	"baseSize": random.uniform(0.1,0.4),
	# 	"baseSize_s": random.uniform(0.2,0.3),
	# 	"splitHeight": random.uniform(0.1,0.4),
	# 	"ratio": random.uniform(0.001,0.01),
	# 	"minRadius": 0.0015,
	# 	"ratioPower": random.uniform(1.0,2.0),
	# 	"downAngle": (random.randint(2,20),random.randint(10,30),random.randint(2,15),random.randint(1,8)),
	# 	"downAngleV": (random.randint(2,10),random.randint(5,20),random.randint(1,5),random.randint(1,3))
	# }

	tree_parameters = {
		"length": (random.uniform(0.3,1.0),random.uniform(0.3,1.0),random.uniform(0.3,1.0),random.uniform(0.3,1.0)),
		"lengthV": (random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3)),
		"segSplits": (random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3)),
		"attractOut": (random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3)),
		"baseSize": random.uniform(0.05,0.45),
		"baseSize_s": random.uniform(0.1,0.4),
		"pruneWidth": random.uniform(0.1,0.4),
		"splitHeight": random.uniform(0.2,0.8),
		"ratio": random.uniform(0.001,0.1),
	}

	# print(tree_parameters.keys())

	'''Randomly Sampled Tree'''
	# bpy.ops.curve.tree_add(bevel=tree_parameters["bevel"], showLeaves=tree_parameters['showLeaves'], seed=tree_parameters["seed"], 
	# 	levels=tree_parameters["levels"], length=tree_parameters["length"], lengthV=tree_parameters["lengthV"], 
	# 	branches=tree_parameters["branches"], curveRes=tree_parameters["curveRes"], curve=tree_parameters["curve"], 
	# 	curveV=tree_parameters["curveV"], curveBack=(0, 20, 0, 0), baseSplits=tree_parameters["baseSplits"], segSplits=tree_parameters["segSplits"], 
	# 	splitByLen=True, rMode='rotate', splitAngle=tree_parameters["splitAngle"], splitAngleV=tree_parameters["splitAngleV"], 
	# 	scale=tree_parameters["scale"], scaleV=tree_parameters["scaleV"], attractUp=tree_parameters["attractUp"], attractOut=tree_parameters["attractOut"], shape='4', shapeS='4', 
	# 	customShape=(0.5, 1, 0.3, 0.5), branchDist=tree_parameters["branchDist"], nrings=0, baseSize=tree_parameters["baseSize"], baseSize_s=tree_parameters["baseSize_s"], 
	# 	splitHeight=tree_parameters["splitHeight"], splitBias=0, ratio=0.025, minRadius=0.0015, closeTip=False, rootFlare=1, autoTaper=True, taper=(1, 1, 1, 1), 
	# 	radiusTweak=(1, 1, 1, 1), ratioPower=tree_parameters["ratioPower"], downAngle=tree_parameters["downAngle"], downAngleV=tree_parameters["downAngleV"], useOldDownAngle=False, 
	# 	useParentAngle=True, rotate=(99.5, 137.5, -60, 140), rotateV=(15, 15, 45, 0), scale0=1, scaleV0=0, leaves=150, 
	# 	leafDownAngle=30, leafDownAngleV=10, leafRotate=137.5, leafRotateV=30, leafScale=0.25, leafScaleX=0.2, 
	# 	leafScaleT=0, leafScaleV=0, leafShape='rect', bend=0, leafangle=0, horzLeaves=False, leafDist='10', bevelRes=1, 
	# 	resU=4, armAnim=False, previewArm=False, leafAnim=False, frameRate=1, loopFrames=0, wind=1, gust=1, gustF=0.075, 
	# 	af1=1, af2=1, af3=4, makeMesh=False, armLevels=2, boneStep=(1, 1, 1, 1))


	'''First Limited Tree (too dense on generation)'''
	# bpy.ops.curve.tree_add(do_update=True, bevel=True, prune=False, showLeaves=False, useArm=False, seed=0, 
	# 	handleType='0', levels=3, length=tree_parameters["length"], lengthV=tree_parameters["lengthV"], taperCrown=0, 
	# 	branches=(0, 60, 30, 10), curveRes=(10, 8, 3, 1), curve=(0, -30, -25, 0), curveV=(100, 80, 80, 0), 
	# 	curveBack=(0, -5, 0, 0), baseSplits=2, segSplits=tree_parameters["segSplits"], splitByLen=True, rMode='rotate', 
	# 	splitAngle=(12, 18, 16, 0), splitAngleV=(2, 2, 0, 0), scale=12, scaleV=2, attractUp=(0, -1, -0.65, 0), 
	# 	attractOut=tree_parameters["attractOut"], shape='8', shapeS='7', customShape=(0.7, 1, 0.3, 0.59), branchDist=tree_parameters["branchDist"], 
	# 	nrings=0, baseSize=tree_parameters["baseSize"], baseSize_s=tree_parameters["baseSize_s"], splitHeight=tree_parameters["splitHeight"], splitBias=0, ratio=tree_parameters["ratio"], minRadius=0.002, 
	# 	closeTip=False, rootFlare=1.15, autoTaper=True, taper=(1, 1, 1, 1), radiusTweak=(1, 1, 1, 1), ratioPower=1.4, 
	# 	downAngle=(90, 60, 50, 45), downAngleV=(0, 25, 30, 10), useOldDownAngle=False, useParentAngle=True, 
	# 	rotate=(137.5, 137.5, 137.5, 137.5), rotateV=(0, 0, 0, 0), scale0=1, scaleV0=0.1, pruneWidth=tree_parameters["pruneWidth"], 
	# 	pruneBase=0.3, pruneWidthPeak=0.5, prunePowerHigh=0.1, prunePowerLow=0.001, pruneRatio=1, leaves=16, 
	# 	leafDownAngle=45, leafDownAngleV=10, leafRotate=137.5, leafRotateV=0, leafScale=0.2, leafScaleX=0.5, 
	# 	leafScaleT=0.2, leafScaleV=0.25, leafShape='hex', bend=0, leafangle=-45, horzLeaves=True, leafDist='6', 
	# 	bevelRes=2, resU=2, armAnim=False, previewArm=False, leafAnim=False, frameRate=1, loopFrames=0, wind=1, 
	# 	gust=1, gustF=0.075, af1=1, af2=1, af3=4, makeMesh=False, armLevels=0, boneStep=(1, 1, 1, 1))

	'''Second limited Tree'''
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
	bpy.data.scenes['Scene'].render.filepath = 'C:/Users/Anthony Hernandez/Desktop/Julies Python Supervised machine learning/renders/Image'+str(i)+'.jpg'
	f = open('C:/Users/Anthony Hernandez/Desktop/Julies Python Supervised machine learning/renders/Image'+str(i)+'.txt', "w+")   # 'r' for reading and 'w' for writing
	f.write(str(tree_parameters))    # Write inside file 
	bpy.ops.render.render( write_still=True )
	bpy.data.objects['tree'].select = False

