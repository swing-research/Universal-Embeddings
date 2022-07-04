# GPU 0 
nohup python -u RandomTree_mixtures.py 0 0 > disp_nb_mixture0.out &
nohup python -u RandomTree_mixtures.py 1 0 > disp_nb_mixture1.out &
# GPU 1 
nohup python -u RandomTree_mixtures.py 2 1 > disp_nb_mixture2.out &
nohup python -u RandomTree_mixtures.py 3 1 > disp_nb_mixture3.out &
# GPU 2 
nohup python -u RandomTree_mixtures.py 4 2 > disp_nb_mixture4.out &
nohup python -u RandomTree_mixtures.py 5 2 > disp_nb_mixture5.out &
# GPU 3
nohup python -u RandomTree_mixtures.py 6 3 > disp_nb_mixture6.out &
nohup python -u RandomTree_mixtures.py 7 3 > disp_nb_mixture7.out &

