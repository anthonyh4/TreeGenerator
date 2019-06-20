import bpy
import random

bpy.context.scene.render.engine = 'CYCLES'

for i in range(1702, 2700):
	if 'tree' in bpy.data.objects.keys():
		bpy.data.objects['tree'].select = True
		bpy.ops.object.delete()
		# bpy.data.objects['leaves'].select = True
		# bpy.ops.object.delete()

	tree_parameters = {
		"bevel": True,
		"showLeaves": False,
		"seed": random.randint(1000,50000),
		"levels": random.randint(3,4),
		"length": (random.uniform(0.3,1.0),random.uniform(0.3,1.0),random.uniform(0.3,1.0),random.uniform(0.3,1.0)),
		"lengthV": (random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3)),
		"branches": (0,random.randint(10,50),random.randint(5,25),random.randint(0,10)),
		"curveRes": (random.randint(5,10),random.randint(15,20),random.randint(5,10),random.randint(1,5)),
		"curve": (random.randint(-5,5),random.randint(-5,5),random.randint(-5,5),0),
		"curveV": (random.randint(0,10),random.randint(0,10),random.randint(0,10),random.randint(0,10)),
		"baseSplits": random.randint(0,2),
		"segSplits": (random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3),random.uniform(0,0.3)),
		"splitByLen": True,
		"splitAngle": (random.randint(2,20),random.randint(10,30),random.randint(2,15),random.randint(1,8)),
		"splitAngleV": (random.randint(1,5),random.randint(7,15),random.randint(1,5),random.randint(1,2)),
		"scale": 20,
		"scaleV": 0,
		"attractUp": (random.uniform(-0.3,0.3),random.uniform(-0.3,0.3),random.uniform(-0.3,0.3),random.uniform(-0.3,0.3)),
		"attractOut": (random.uniform(-0.3,0.3),random.uniform(-0.3,0.3),random.uniform(-0.3,0.3),random.uniform(-0.3,0.3)),
		"branchDist": random.uniform(1.0,2.0),
		"baseSize": random.uniform(0.1,0.4),
		"baseSize_s": random.uniform(0.2,0.3),
		"splitHeight": random.uniform(0.1,0.4),
		"ratio": random.uniform(0.001,0.01),
		"minRadius": 0.0015,
		"ratioPower": random.uniform(1.0,2.0),
		"downAngle": (random.randint(2,20),random.randint(10,30),random.randint(2,15),random.randint(1,8)),
		"downAngleV": (random.randint(2,10),random.randint(5,20),random.randint(1,5),random.randint(1,3))
	}


	# TreeSeed = random.randint(1000,5000)

	bpy.ops.curve.tree_add(bevel=tree_parameters["bevel"], showLeaves=tree_parameters['showLeaves'], seed=tree_parameters["seed"], 
		levels=tree_parameters["levels"], length=tree_parameters["length"], lengthV=tree_parameters["lengthV"], 
		branches=tree_parameters["branches"], curveRes=tree_parameters["curveRes"], curve=tree_parameters["curve"], 
		curveV=tree_parameters["curveV"], curveBack=(0, 20, 0, 0), baseSplits=tree_parameters["baseSplits"], segSplits=tree_parameters["segSplits"], 
		splitByLen=True, rMode='rotate', splitAngle=tree_parameters["splitAngle"], splitAngleV=tree_parameters["splitAngleV"], 
		scale=tree_parameters["scale"], scaleV=tree_parameters["scaleV"], attractUp=tree_parameters["attractUp"], attractOut=tree_parameters["attractOut"], shape='4', shapeS='4', 
		customShape=(0.5, 1, 0.3, 0.5), branchDist=tree_parameters["branchDist"], nrings=0, baseSize=tree_parameters["baseSize"], baseSize_s=tree_parameters["baseSize_s"], 
		splitHeight=tree_parameters["splitHeight"], splitBias=0, ratio=0.025, minRadius=0.0015, closeTip=False, rootFlare=1, autoTaper=True, taper=(1, 1, 1, 1), 
		radiusTweak=(1, 1, 1, 1), ratioPower=tree_parameters["ratioPower"], downAngle=tree_parameters["downAngle"], downAngleV=tree_parameters["downAngleV"], useOldDownAngle=False, 
		useParentAngle=True, rotate=(99.5, 137.5, -60, 140), rotateV=(15, 15, 45, 0), scale0=1, scaleV0=0, leaves=150, 
		leafDownAngle=30, leafDownAngleV=10, leafRotate=137.5, leafRotateV=30, leafScale=0.25, leafScaleX=0.2, 
		leafScaleT=0, leafScaleV=0, leafShape='rect', bend=0, leafangle=0, horzLeaves=False, leafDist='10', bevelRes=1, 
		resU=4, armAnim=False, previewArm=False, leafAnim=False, frameRate=1, loopFrames=0, wind=1, gust=1, gustF=0.075, 
		af1=1, af2=1, af3=4, makeMesh=False, armLevels=2, boneStep=(1, 1, 1, 1))

	bpy.data.objects['tree'].select = True
	
	##bpy.ops.view3d.viewnumpad(type='FRONT', align_active=True)
	##bpy.ops.view3d.camera_to_view()
	bpy.ops.view3d.camera_to_view_selected()

	bpy.data.scenes['Scene'].render.filepath = 'C:/Users/Anthony Hernandez/Desktop/Julies Python Supervised machine learning/renders/Image'+str(i)+'.jpg'
	f = open('C:/Users/Anthony Hernandez/Desktop/Julies Python Supervised machine learning/renders/Image'+str(i)+'.txt', "w+")   # 'r' for reading and 'w' for writing
	f.write(str(tree_parameters))    # Write inside file 
	bpy.ops.render.render( write_still=True )
	bpy.data.objects['tree'].select = False

