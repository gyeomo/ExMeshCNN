import pymeshlab as ml
import numpy as np
import os


target_num = 500
dataset_name = "cubes"

def set_header(text_file_path):
    new_text_content = ''
    with open(text_file_path, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if l.startswith('v'): new_text_content += l
            elif l.startswith('f'): new_text_content += l

    with open(text_file_path,'w') as f:
        f.write(new_text_content)

dataset = "./{}_{}".format(dataset_name,str(target_num))
database = os.path.join(os.getcwd(), dataset_name)
modes = ["train", "test"]
class_names = os.listdir(database)
for class_name in class_names:
    for mode in modes:
        path1 = os.path.join(database, class_name, mode)
        obj_names = os.listdir(path1)
        for obj in obj_names:
            try:
                path2 = os.path.join(path1, obj)
                set_header(path2)
                ms = ml.MeshSet()
                ms.load_new_mesh(path2)
                m = ms.current_mesh()
                iter_num = 0
                if m.face_number() < target_num:
                    while (ms.current_mesh().face_number() < target_num):
                        if iter_num == 100:
                            break
                        ms.apply_filter('subdivision_surfaces_midpoint', iterations=1)
                        iter_num += 1

                TARGET=target_num

                #Estimate number of faces to have 100+10000 vertex using Euler
                numFaces = 100 + 2*TARGET

                #Simplify the mesh. Only first simplification will be agressive
                iter_num = 0
                while (ms.current_mesh().face_number() > TARGET):
                    if iter_num == 100:
                        break
                    ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preserveboundary = True, preservenormal=True, preservetopology = True)
                    numFaces = numFaces - (ms.current_mesh().face_number() - TARGET)
                    iter_num += 1

                ms.apply_filter('repair_non_manifold_edges_by_removing_faces')
                ms.apply_filter('repair_non_manifold_vertices_by_splitting', vertdispratio = 0)
                m = ms.current_mesh()

                save_path = os.path.join(dataset, class_name, mode)
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, obj)
                ms.save_current_mesh(save_path)
            except:
                continue
