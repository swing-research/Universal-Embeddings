# GPU 1
nohup python -u Sphere_dimension_Hyperbolic.py 0 0 > disp_sphere_Euclidean0.out &
nohup python -u Sphere_dimension_Hyperbolic.py 1 0 > disp_sphere_Euclidean1.out &
# # GPU 2
nohup python -u Sphere_dimension_Hyperbolic.py 2 1 > disp_sphere_Euclidean2.out &
nohup python -u Sphere_dimension_Hyperbolic.py 3 1 > disp_sphere_Euclidean3.out &
