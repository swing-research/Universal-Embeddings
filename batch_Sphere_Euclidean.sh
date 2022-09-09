# GPU 1
nohup python -u Sphere_dimension_Euclidean.py 0 2 > disp_sphere_Euclidean0.out &
nohup python -u Sphere_dimension_Euclidean.py 1 2 > disp_sphere_Euclidean1.out &
# # GPU 2
nohup python -u Sphere_dimension_Euclidean.py 2 3 > disp_sphere_Euclidean2.out &
nohup python -u Sphere_dimension_Euclidean.py 3 3 > disp_sphere_Euclidean3.out &
