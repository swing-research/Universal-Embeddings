# GPU 0 
nohup python -u Sphere_dimension_MG.py 0 0 > disp_sphere_MG0.out &
nohup python -u Sphere_dimension_MG.py 1 0 > disp_sphere_MG1.out &
# GPU 1 
nohup python -u Sphere_dimension_MG.py 2 1 > disp_sphere_MG2.out &
nohup python -u Sphere_dimension_MG.py 3 1 > disp_sphere_MG3.out &